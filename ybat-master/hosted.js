(() => {
    "use strict";
    const $ = (id) => document.getElementById(id);
    const state = { session: null, sessionEpoch: 0, datasets: [], images: [], selected: new Map(), users: [], view: "confirmed", cursor: null, total: null, sequence: 0, workspaceSequence: 0, userSequence: 0, classSequence: 0, submissionSequence: 0, countSequence: 0, classDataset: null, submissions: [], submissionImages: new Set(), selectionAnchor: null, detail: null, filters: {q: "", incoming: "incoming", tranche: "", annotator: ""} };
    const op = () => crypto.randomUUID();
    const node = (tag, text, className) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; };
    const count = (value, one, many = `${one}s`) => `${Number(value || 0).toLocaleString()} ${value === 1 ? one : many}`;
    const VIEWS = ["confirmed", "incoming", "tranches", "submitted"];
    const INCOMING_FILTERS = ["incoming", "unallocated", "needs_decision", "with_labels", "without_labels"];
    // Select all matching and the bulk actions work on at most this many images;
    // requests go to the server in chunks, one receipt per image.
    const SELECTION_LIMIT = 5000, BULK_CHUNK = 250;
    function notice(message, error = false) {
        for (const element of [$("notice"), ...document.querySelectorAll("dialog[open] [data-dialog-notice]")]) { element.textContent = message; element.dataset.error = String(error); element.hidden = !message; }
    }
    const errorMessages = {tranche_revision_changed: "The tranche changed. Refresh it before applying this change.", tranche_capacity_exceeded: "A tranche can hold at most 10,000 unfinished images. Create another tranche for the remaining images.", image_left_queue: "That image left this queue. Reopen the tranche to continue.", tranche_has_pending_review: "Review the submitted images first, or move selected unfinished images to another tranche.", tranche_has_unfinished_members: "Finish or remove the remaining work before closing this tranche.", image_not_assignable: "An image is no longer available in Incoming. Refresh the selection before trying again.", dataset_revision_changed: "The dataset changed. Refresh the preview before applying this change.", split_group_conflict: "These images share a capture group. Preview the complete group before changing its split.", editor_lease_conflict: "An image is open for editing. Close its editing tab before changing this selection.", last_administrator: "Keep at least one active administrator.", username_exists: "That username is already used, possibly by a removed account. Choose a different username.", class_version_changed: "The class list changed. Refresh classes and review the current names before trying again.", class_request_already_reviewed: "This request has already been reviewed. Refresh classes to see the result.", class_name_exists: "That class name already exists.", user_changed: "This account changed. Refresh People before applying your change.", installation_changed: "The server changed. Sign in again before continuing.",
        invalid_or_expired_link: "This link has expired or was already used. Ask your administrator for a new link.", invalid_credentials: "Username or password is incorrect.", login_rate_limited: "Too many sign-in attempts. Wait 15 minutes, then try again.", login_required: "Sign in to continue.", actor_inactive: "This account can no longer sign in. Ask an administrator.", csrf_rejected: "This page is out of date. Reload it and sign in again.", admin_required: "Only an administrator can do this.", administrator_required: "Only an administrator can do this.",
        annotation_version_changed: "The image changed since this page loaded. Refresh before trying again.", ownership_changed: "The image changed hands since this page loaded. Refresh before trying again.", editor_lease_expired: "The editing session expired. Reopen the image to continue.", annotation_not_complete: "Mark the image as checked before approving it.", empty_annotation_requires_negative_review: "An image without boxes must be marked as having no target objects before approval.", image_not_approvable: "This image is no longer waiting for approval. Refresh to see its state.", image_not_returnable: "This image is no longer waiting for review.", image_not_submittable: "This image cannot be submitted in its current state.", image_not_overridable: "This image is no longer assigned to anyone.", image_not_found: "That image is no longer available.", image_not_editable: "You cannot edit this image.", image_dataset_mismatch: "That image belongs to another dataset.", image_owned_by_annotator: "This image is assigned to an annotator. Recall it first.", submission_changed: "The submission changed. Refresh before reviewing it.",
        dataset_not_found: "That dataset is not available.", dataset_still_archived: "Restore the dataset before restoring its images.", duplicate_image_archived: "This image already exists in the dataset but is archived. Restore it from Export & recovery.", archive_already_restored: "This item was already restored.", archive_membership_changed: "The archived images changed. Refresh Export & recovery.", archive_version_changed: "The archive changed. Refresh Export & recovery.",
        tranche_not_found: "That tranche is not available.", tranche_closed: "This tranche is closed. Reopen it first.", tranche_not_active: "This tranche is not active.", tranche_not_draft: "This tranche is no longer a Draft.", tranche_membership_changed: "The tranche's images changed. Refresh it before trying again.", tranche_member_not_movable: "Some selected images cannot be moved in their current state.", invalid_tranche_destination: "Choose another open tranche of the same dataset.", annotator_unavailable: "That annotator's account is not available.", user_has_open_work: "This person still has open images. Review, recall or move them first.", user_has_open_tranches: "This person still has open tranches. Reassign or close them first.", user_not_found: "That account is not available.", user_not_available: "That account is not available.",
        invalid_class: "A box uses a class that is not in this dataset.", import_classes_missing: "The labels use classes that are not in this dataset. Add the classes first.", ambiguous_import_labelmap: "The labelmap is ambiguous. Select exactly one labelmap file.", bbox_out_of_bounds: "A box lies completely outside the image.", operation_id_reused: "This request was already used for a different change. Refresh and try again.", queue_scope_changed: "The work list changed. Refresh to continue.", queue_scope_mismatch: "That tranche does not belong to this dataset.", invalid_queue: "Choose one of the listed views.", invalid_search: "Use at most 200 characters in the search.",
        image_needs_decision: "This image needs a decision about its imported labels. Open it to choose.", image_has_no_labels: "This image has no labels; confirm it as background instead.", image_has_labels: "This image has labels; approve it with its labels instead.", invalid_bulk_targets: "The selection repeated an image. Refresh and select again.", purge_retention_not_elapsed: "These files are still inside their recovery period.", purge_references_changed: "Storage references changed. Scan again before cleaning up."};
    const humanize = (code) => { const text = String(code).replaceAll("_", " "); return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`; };
    const errorText = (code, detail) => errorMessages[code] || (typeof detail === "string" ? detail : code ? humanize(code) : "");
    async function api(path, options = {}) {
        const authority = state.session, epoch = state.sessionEpoch;
        const headers = new Headers(options.headers);
        if (authority) { headers.set("x-tator-csrf", authority.csrf); headers.set("x-tator-instance", authority.instance_id); }
        const operationId = options.body instanceof FormData ? options.body.get("operation_id") : options.body?.operation_id || options.body?.context?.operation_id;
        let body = options.body;
        if (body && !(body instanceof FormData)) { headers.set("content-type", "application/json"); body = JSON.stringify(body); }
        for (let attempt = 0; ; attempt++) {
            let response, data;
            try {
                response = await fetch(`/api/hosted${path}`, { ...options, body, headers, credentials: "same-origin", cache: "no-store" });
                data = await response.json();
            } catch (cause) {
                if (state.session !== authority || state.sessionEpoch !== epoch) throw Object.assign(new Error("This account view has changed."), { stale: true });
                if (operationId && attempt === 0) continue; // Same exact operation and payload; the server replays its receipt.
                const mutation = options.method && !["GET", "HEAD"].includes(options.method);
                throw new Error(mutation ? "The connection ended before confirmation. Refresh to check whether the change was saved before trying again." : "Could not refresh from the server. Check the connection and try again.", { cause });
            }
            if (state.session !== authority || state.sessionEpoch !== epoch) throw Object.assign(new Error("This account view has changed."), { stale: true });
            if (!response.ok) {
                const error = new Error(data.commit_outcome === "committed" && data.result_restricted === true
                    ? "The change was saved, but your account can no longer view its result. Refresh the workspace to see what you can access."
                    : errorText(data.error, data.detail) || `Request failed (${response.status})`);
                error.status = response.status; error.code = data.error; error.detail = data.detail;
                for (const name of ["commit_outcome", "operation_id", "result_restricted"]) error[name] = data[name];
                if ((response.status === 401 || ["installation_changed", "csrf_rejected"].includes(data.error)) && authority) loginView();
                throw error;
            }
            return data;
        }
    }
    async function action(button, work) {
        const controls = button.form ? Array.from(button.form.elements) : [button], disabled = controls.map((control) => control.disabled);
        controls.forEach((control) => { control.disabled = true; });
        try { await work(); } catch (error) { if (!error.stale) notice(error.message, true); }
        finally { controls.forEach((control, index) => { control.disabled = disabled[index]; }); }
    }
    async function afterCommit(message, refresh) {
        try { await refresh(); notice(message); }
        catch (error) { if (!error.stale) notice(`${message} The view could not refresh; use Refresh to load the saved state.`, true); }
    }
    // Light or dark; double-click the button for the Pip-Boy theme, where a click switches green and amber.
    // The keys are the annotation canvas and workbench ones, so the choice follows the person across the app.
    (() => {
        const read = key => { try { return localStorage.getItem(key); } catch (_) { return null; } };
        const write = (key, value) => { try { localStorage.setItem(key, value); } catch (_) {} };
        const saved = read("tator.themeMode"), older = read("tator.hosted.theme");
        let mode = ["light", "dark", "pipboy"].includes(saved) ? saved : ["light", "dark"].includes(older) ? older : matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        let classic = ["light", "dark"].includes(read("tator.classicThemeMode")) ? read("tator.classicThemeMode") : mode === "dark" ? "dark" : "light";
        let accent = read("tator.pipboyAccent") === "amber" ? "amber" : "green", clickTimer = null;
        function apply(next, persist) {
            mode = next; if (mode !== "pipboy") classic = mode;
            const root = document.documentElement, button = $("theme"), pip = mode === "pipboy";
            root.dataset.theme = mode; if (pip) root.dataset.pipboyAccent = accent; else delete root.dataset.pipboyAccent;
            button.textContent = pip ? `Pip ${accent === "amber" ? "Amber" : "Green"}` : mode === "dark" ? "Light mode" : "Dark mode";
            button.title = pip ? `Switch Pip-Boy color to ${accent === "amber" ? "green" : "amber"}; double-click to leave Pip-Boy` : "";
            if (!persist) return;
            write("tator.themeMode", mode); write("tator.darkMode", mode === "dark" ? "1" : "0"); write("tator.pipboyAccent", accent);
            if (!pip) write("tator.classicThemeMode", mode);
        }
        $("theme").onclick = () => {
            clearTimeout(clickTimer);
            clickTimer = setTimeout(() => { clickTimer = null; if (mode === "pipboy") accent = accent === "amber" ? "green" : "amber"; apply(mode === "pipboy" ? mode : mode === "dark" ? "light" : "dark", true); }, 320);
        };
        $("theme").ondblclick = () => { clearTimeout(clickTimer); clickTimer = null; apply(mode === "pipboy" ? classic : "pipboy", true); };
        apply(mode, false);
    })();
    // A page opened while signed out (?next=) opens again after signing in.
    // Only this server's own pages qualify, so the link cannot send anyone elsewhere.
    function nextPage() {
        const raw = new URLSearchParams(location.search).get("next");
        if (!raw || !raw.startsWith("/") || raw.startsWith("//") || raw.includes("\\")) return null;
        try { const url = new URL(raw, location.origin); return url.origin === location.origin && ["/annotate", "/pruning", "/workbench"].includes(url.pathname) ? url.pathname + url.search : null; }
        catch (_) { return null; }
    }
    function announceSession() { try { localStorage.setItem("tator.hosted.session-event", crypto.randomUUID()); } catch (_) {} }
    window.addEventListener("storage", event => { if (event.key === "tator.hosted.session-event" && state.session) { loginView(); notice("The signed-in account changed in another tab. Sign in or reload to continue."); } });
    function loginView() {
        state.session = null; state.sessionEpoch++; state.sequence++; state.workspaceSequence++; state.userSequence++; state.classSequence++; state.submissionSequence++; state.countSequence++;
        state.images = []; state.datasets = []; state.users = []; state.submissions = []; state.classDataset = null; state.detail = null; state.selected.clear();
        for (const id of ["imageGrid", "userList", "classList", "classRequestList", "submissionList", "datasetChoice", "assignee"]) $(id).replaceChildren();
        document.querySelectorAll("dialog[open]").forEach((dialog) => dialog.close());
        $("invitationLink").value = ""; $("invitation").hidden = true;
        $("workspace").hidden = true; $("loginPanel").hidden = false; $("logout").hidden = true; $("people").hidden = true; $("accountName").textContent = "";
        window.dispatchEvent(new Event("tator:hosted-session"));
    }
    function invitationView() {
        const link = new URLSearchParams(location.hash.slice(1));
        if (!link.has("invite")) return false;
        loginView(); $("loginPanel").hidden = true; $("invitePanel").hidden = false; notice("");
        const username = (link.get("user") || "").trim().slice(0, 64);
        $("inviteUsername").value = username; $("inviteUsernameLabel").hidden = !username;
        $("inviteGreeting").textContent = username ? `Choose a password for the username ${username}. You sign in with this username.` : "Choose a password for the username your administrator sent you.";
        return true;
    }

    // ---- Route: the URL names the dataset, view and filters; Back and Forward restore them.
    function routeQuery(tranche = null) {
        const query = new URLSearchParams(), dataset = $("datasetChoice").value;
        if (dataset) query.set("dataset", dataset);
        query.set("view", state.view);
        if (["tranches", "mine"].includes(state.view)) { if (tranche) query.set("tranche", tranche); return query; }
        if (state.filters.q) query.set("q", state.filters.q);
        if (state.view === "incoming" && state.filters.incoming !== "incoming") query.set("filter", state.filters.incoming);
        if (state.view === "submitted") { if (state.filters.tranche) query.set("tranche", state.filters.tranche); if (state.filters.annotator) query.set("annotator", state.filters.annotator); }
        return query;
    }
    function writeRoute(mode, tranche = null) {
        const query = routeQuery(tranche), current = new URLSearchParams(location.search);
        // Links that open a dialog or return to an image keep their parameter until handled.
        if (mode === "replace") for (const key of ["tool", "analysis", "panel", "at"]) if (current.has(key)) query.set(key, current.get(key));
        const url = `/?${query}`;
        if (url !== location.pathname + location.search) history[mode === "push" ? "pushState" : "replaceState"](null, "", url);
    }
    function dropRouteParams(...keys) {
        const url = new URL(location.href); let changed = false;
        for (const key of keys) if (url.searchParams.has(key)) { url.searchParams.delete(key); changed = true; }
        if (changed) history.replaceState(history.state, "", url.pathname + url.search);
    }
    function syncFilterControls() {
        $("imageSearch").value = state.filters.q; $("incomingFilter").value = state.filters.incoming;
        $("reviewAnnotator").value = state.filters.annotator; if ([...$("reviewTranche").options].some(option => option.value === state.filters.tranche)) $("reviewTranche").value = state.filters.tranche;
    }
    function resetFilters() { state.filters = {q: "", incoming: "incoming", tranche: "", annotator: ""}; syncFilterControls(); }
    async function applyRoute(route = new URLSearchParams(location.search)) {
        const isAdmin = state.session?.user.role === "admin", previous = $("datasetChoice").value;
        const wanted = route.get("dataset");
        if (wanted && state.datasets.some(row => row.id === wanted)) $("datasetChoice").value = wanted;
        let next = isAdmin ? route.get("view") : "mine";
        if (isAdmin && !VIEWS.includes(next)) next = route.get("tranche") ? "tranches" : "confirmed";
        const filter = route.get("filter");
        state.filters = {q: (route.get("q") || "").trim().slice(0, 200), incoming: INCOMING_FILTERS.includes(filter) ? filter : "incoming",
            tranche: next === "submitted" ? route.get("tranche") || "" : "", annotator: next === "submitted" ? route.get("annotator") || "" : ""};
        syncFilterControls();
        if ($("datasetChoice").value !== previous) window.dispatchEvent(new Event("tator:hosted-dataset"));
        // An annotator who opens an administrator's link sees their own work, and is told why.
        const adminLink = !isAdmin && ((route.get("view") && route.get("view") !== "mine") || ["tool", "analysis", "panel"].some(key => route.has(key)));
        if (adminLink) dropRouteParams("tool", "analysis", "panel");
        await view(next, {route: "replace", tranche: ["tranches", "mine"].includes(next) ? route.get("tranche") : null});
        if (adminLink) notice("That link opens an administrator view, so your own work is shown instead.");
        await revealImage(route.get("at"));
    }
    async function navigate(next) {
        notice("");
        state.view = next; writeRoute("push");
        await view(next);
    }
    window.addEventListener("popstate", () => {
        if (!state.session || $("workspace").hidden || new URLSearchParams(location.hash.slice(1)).has("invite")) return;
        document.querySelectorAll("dialog[open]").forEach(dialog => dialog.close());
        notice(""); applyRoute().catch(error => { if (!error.stale) notice(error.message, true); });
    });
    async function revealImage(id) {
        if (!id) return;
        const find = () => document.querySelector(`#imageGrid [data-image-id="${CSS.escape(id)}"], #tranchePanel [data-image-id="${CSS.escape(id)}"]`);
        // Back from the canvas: page forward until the image's card is shown (bounded).
        for (let page = 0; page < 20 && !find() && state.cursor && !$("imageGrid").hidden; page++) {
            const before = state.sequence; await loadImages(true);
            if (state.sequence !== before + 1) break;
        }
        dropRouteParams("at");
        const card = find(); if (!card) return;
        card.scrollIntoView({block: "center"}); card.classList.add("hosted-highlight");
        setTimeout(() => card.classList.remove("hosted-highlight"), 2600);
    }

    // ---- Counts, dataset name and import report beside the views.
    const datasetName = () => state.datasets.find(row => row.id === $("datasetChoice").value)?.name || "";
    function incomingCount(counts) { return ["unassigned", "assigned", "returned", "submitted"].reduce((sum, key) => sum + (counts?.[key] || 0), 0); }
    function drawTabs() {
        const counts = state.detail?.id === $("datasetChoice").value ? state.detail.counts || {} : null;
        for (const button of $("views").children) {
            const badge = button.querySelector(".view-count"); if (!badge) continue;
            const value = !counts ? null : {confirmed: counts.confirmed || 0, incoming: incomingCount(counts), submitted: counts.submitted || 0}[button.dataset.view];
            badge.textContent = value === null || value === undefined ? "" : button.dataset.view === "submitted" ? `${value.toLocaleString()} waiting` : value.toLocaleString();
            $(`${button.id}Count`).textContent = value === null || value === undefined ? "" : button.dataset.view === "submitted" ? `${count(value, "image")} waiting for review` : count(value, "image");
        }
    }
    function drawTitle() {
        const name = datasetName(), admin = state.session?.user.role === "admin";
        $("workspaceEyebrow").textContent = name || (admin ? "No dataset yet" : "");
        document.title = name ? `${name} · ${$("workspaceTitle").textContent} · Tator` : "Tator · Dataset workspace";
    }
    function drawImportReport() {
        const box = $("importReport"), dataset = $("datasetChoice").value, detail = state.detail;
        box.replaceChildren(); box.hidden = true;
        const report = detail?.id === dataset && Array.isArray(detail.import_reports) ? detail.import_reports[0] : null;
        if (!report || typeof report !== "object" || state.session?.user.role !== "admin" || !["confirmed", "incoming"].includes(state.view)) return;
        const number = value => Array.isArray(value) ? value.length : Number.isFinite(value) ? value : null;
        const images = number(report.images), confirmed = number(report.confirmed), incoming = number(report.incoming), parts = [];
        if (images !== null) parts.push(`${count(images, "image")}${confirmed !== null && incoming !== null ? ` (${confirmed.toLocaleString()} confirmed, ${incoming.toLocaleString()} Incoming)` : ""}`);
        if (number(report.without_label_file)) parts.push(`${count(number(report.without_label_file), "image")} without a label file`);
        if (number(report.clipped_rows)) parts.push(`${count(number(report.clipped_rows), "box", "boxes")} clipped to the image`);
        if (number(report.dropped_rows)) parts.push(`${count(number(report.dropped_rows), "empty row")} dropped`);
        if (number(report.duplicate_copies_removed)) parts.push(`${count(number(report.duplicate_copies_removed), "duplicate copy", "duplicate copies")} removed`);
        const decisions = number(report.review_required) ?? number(report.duplicate_conflicts);
        if (decisions) parts.push(`${count(decisions, "image")} ${decisions === 1 ? "needs" : "need"} a decision`);
        const source = String(report.source || "").split(/[\\/]/).filter(Boolean).at(-1) || "another dataset";
        const when = Number.isFinite(report.created_at) ? ` on ${new Date(report.created_at * 1000).toLocaleDateString()}` : "";
        box.append(node("span", `Imported from ${source}${when}${parts.length ? ": " : ""}${parts.join(" · ")}`));
        if (decisions) {
            const link = node("a", decisions === 1 ? "Show the image that needs a decision" : "Show the images that need a decision");
            link.href = `/?${new URLSearchParams({dataset, view: "incoming", filter: "needs_decision"})}`;
            link.onclick = event => { event.preventDefault(); state.filters = {...state.filters, q: "", incoming: "needs_decision"}; syncFilterControls(); navigate("incoming").catch(error => { if (!error.stale) notice(error.message, true); }); };
            box.append(document.createTextNode(" · "), link);
        }
        box.hidden = false;
    }
    async function refreshCounts() {
        const id = $("datasetChoice").value, sequence = ++state.countSequence;
        if (!id || state.session?.user.role !== "admin") { state.detail = null; drawTabs(); drawImportReport(); return; }
        try {
            const detail = await api(`/datasets/${encodeURIComponent(id)}`);
            if (sequence !== state.countSequence || id !== $("datasetChoice").value) return;
            state.detail = detail; drawTabs(); drawImportReport();
        } catch (error) { if (!error.stale && sequence === state.countSequence) { state.detail = null; drawTabs(); drawImportReport(); } }
    }
    function syncDatasetActions() {
        const isAdmin = state.session?.user.role === "admin", none = !state.datasets.length;
        const hint = isAdmin ? "Create a dataset first." : "Available once work is assigned to you.";
        for (const id of ["classes", "hostedGuidance", "import", "recallAssigned"]) {
            const control = $(id); if (!control) continue;
            const metadata = id === "classes" && Boolean(state.datasets.find(d => d.id === $("datasetChoice").value)?.metadata_only);
            control.disabled = none || metadata; control.title = none ? hint : metadata ? "Your work in this dataset is finished; its classes are no longer shared with you." : "";
        }
        $("createDataset").classList.toggle("primary", isAdmin && none);
        $("datasetChoiceLabel").hidden = none; $("views").hidden = none && isAdmin;
    }

    // ---- Images, filters and selection.
    function bulkCounts(rows) {
        let labelled = 0, empty = 0;
        for (const row of rows) if (state.view === "incoming" && !row.needs_decision) { if (row.object_count > 0) labelled++; else empty++; }
        return {labelled, empty};
    }
    function selection() {
        const size = state.selected.size, rows = [...state.selected.values()], incoming = state.view === "incoming", bulk = bulkCounts(rows);
        $("selectionBar").hidden = size === 0;
        $("selectionCount").textContent = `${count(size, "image")} selected`;
        $("selectionBreakdown").textContent = incoming && size ? `${bulk.labelled.toLocaleString()} with labels · ${bulk.empty.toLocaleString()} without${size - bulk.labelled - bulk.empty ? ` · ${(size - bulk.labelled - bulk.empty).toLocaleString()} need a decision` : ""}` : "";
        $("assign").hidden = !incoming; $("assigneeLabel").hidden = true; $("assign").textContent = "Create tranche from selection";
        $("approveIncoming").hidden = !incoming || !bulk.labelled; $("approveIncoming").textContent = `Approve selected (${bulk.labelled.toLocaleString()})`;
        $("confirmBackground").hidden = !incoming || !bulk.empty; $("confirmBackground").textContent = `Confirm as background (${bulk.empty.toLocaleString()})`;
        $("approveSelected").hidden = state.view !== "submitted";
        $("assign").disabled = size === 0;
        window.dispatchEvent(new Event("tator:hosted-selection"));
    }
    function clearSelection() { state.selected.clear(); state.selectionAnchor = null; document.querySelectorAll(".hosted-card input").forEach((input) => { input.checked = false; }); selection(); }
    function invalidateImages() {
        ++state.sequence; clearSelection(); state.images = []; state.cursor = null; state.total = null;
        $("imageGrid").replaceChildren(); $("imageMatchCount").textContent = ""; $("more").hidden = true; $("moreAll").hidden = true;
        return {sequence: state.sequence, dataset: $("datasetChoice").value, view: state.view};
    }
    function currentImages(capture) { return capture.sequence === state.sequence && capture.dataset === $("datasetChoice").value && capture.view === state.view; }
    function selectable(image) { return state.view === "incoming" ? image.state === "unassigned" && !image.tranche_id : state.view === "submitted" ? image.state === "submitted" : image.state === "confirmed"; }
    function queueQuery(limit = 100) {
        const q = new URLSearchParams({dataset_id: $("datasetChoice").value, limit: String(limit), include_total: "true", queue: state.view === "incoming" ? state.filters.incoming : state.view === "submitted" ? "review" : "confirmed"});
        if (state.filters.q) q.set("q", state.filters.q);
        if (state.view === "submitted") { if (state.filters.tranche) q.set("tranche_id", state.filters.tranche); if (state.filters.annotator) q.set("annotator_id", state.filters.annotator); }
        return q;
    }
    function syncChecks() { document.querySelectorAll("#imageGrid input[data-image]").forEach(input => { input.checked = state.selected.has(input.dataset.image); }); selection(); }
    async function selectMatching() {
        const authority = state.session, captured = {sequence: state.sequence, dataset: $("datasetChoice").value, view: state.view}, filters = JSON.stringify(state.filters);
        const stale = () => authority !== state.session || !currentImages(captured) || filters !== JSON.stringify(state.filters);
        const rows = []; let cursor = null, total = 0;
        do {
            const query = queueQuery(500);
            if (state.view === "incoming") query.set("not_assigned", "true");
            if (cursor) query.set("cursor", cursor);
            const page = await api(`/queue?${query}`);
            if (stale()) return;
            total = page.total; if (total > SELECTION_LIMIT) throw new Error(`${total.toLocaleString()} eligible images match. Narrow the search or filter to at most ${SELECTION_LIMIT.toLocaleString()} for one selection.`);
            rows.push(...page.images); cursor = page.next_cursor;
            if (cursor) $("imageMatchCount").textContent = `Collecting ${rows.length.toLocaleString()} of ${total.toLocaleString()} eligible images…`;
        } while (cursor && rows.length < SELECTION_LIMIT);
        if (!window.confirm(`Select all ${count(rows.length, "eligible matching image")}, including those not yet displayed?`)) { drawMatchCount(); return; }
        if (stale()) return;
        state.selected = new Map(rows.map(image => [image.id, image])); syncChecks(); drawMatchCount();
    }
    async function prepareFilters() {
        const isAdmin = state.session?.user.role === "admin", flat = !["tranches", "mine"].includes(state.view);
        $("imageFilters").hidden = !isAdmin || !flat || !$("datasetChoice").value;
        $("incomingFilterLabel").hidden = state.view !== "incoming";
        $("reviewTrancheLabel").hidden = $("reviewAnnotatorLabel").hidden = state.view !== "submitted";
        if (!isAdmin || state.view !== "submitted") return;
        $("reviewAnnotator").replaceChildren(new Option("All annotators", ""), ...state.users.map(user => new Option(personLabel(user), user.id)));
        $("reviewAnnotator").value = state.filters.annotator;
        const sequence = state.sequence, dataset = $("datasetChoice").value, tranches = []; let cursor = null;
        if (!dataset) return;
        do { const q = new URLSearchParams({dataset_id: dataset, limit: "100"}); if (cursor) q.set("cursor", cursor); const page = await api(`/tranches?${q}`); if (sequence !== state.sequence || dataset !== $("datasetChoice").value) return; tranches.push(...page.tranches); cursor = page.next_cursor; } while (cursor);
        $("reviewTranche").replaceChildren(new Option("All tranches", ""), ...tranches.map(t => new Option(t.name, t.id)));
        if (!tranches.some(t => t.id === state.filters.tranche)) state.filters.tranche = "";
        $("reviewTranche").value = state.filters.tranche;
    }
    async function refreshUsers() {
        const sequence = ++state.userSequence, previous = $("assignee").value;
        const result = await api("/users");
        if (sequence !== state.userSequence) return;
        state.users = result.users;
        $("assignee").replaceChildren();
        for (const user of state.users.filter((row) => row.role === "annotator" && !row.disabled)) { const option = node("option", personLabel(user)); option.value = user.id; $("assignee").append(option); }
        if (state.users.some((row) => row.id === previous && row.role === "annotator" && !row.disabled)) $("assignee").value = previous;
        selection();
        drawUsers();
    }
    const decisionReasons = {duplicate_labels_differ: "duplicate copies have different labels"};
    function stateLabel(image) {
        if (image.state === "assigned") return `Assigned to ${image.assignee_name || "an annotator"}`;
        return { confirmed: "Confirmed", returned: "Returned for changes", submitted: "Waiting for review", unassigned: "Not assigned" }[image.state] || image.state;
    }
    function cardLink(image) {
        // Incoming includes every intake state; let the server derive each
        // image's actual tranche scope instead of widening its canvas queue.
        const route = new URLSearchParams({image: image.id});
        if (state.view !== "incoming") route.set("queue", state.view === "submitted" ? "review" : "confirmed");
        route.set("dataset", $("datasetChoice").value); route.set("view", state.view);
        if (state.view === "submitted" && state.filters.tranche) route.set("tranche", state.filters.tranche);
        if (state.filters.q) route.set("q", state.filters.q);
        if (state.view === "incoming" && state.filters.incoming !== "incoming") route.set("filter", state.filters.incoming);
        if (state.view === "submitted" && state.filters.annotator) route.set("annotator", state.filters.annotator);
        return `/annotate?${route}`;
    }
    function drawCard(image) {
        const card = node("article", undefined, "hosted-card"); card.dataset.imageId = image.id;
        const link = node("a"); link.href = cardLink(image);
        const photo = node("img"); photo.decoding = "async"; photo.onerror = () => { photo.onerror = null; photo.src = `/api/hosted/images/${encodeURIComponent(image.id)}/media`; }; photo.src = `/api/hosted/images/${encodeURIComponent(image.id)}/thumbnail`; photo.alt = image.filename; photo.loading = "lazy";
        const info = node("div", undefined, "card-info"); info.append(node("strong", image.filename));
        const status = node("small", `${stateLabel(image)}${image.negative ? " · No target objects" : ""}`);
        status.dataset.state = image.state; if (image.state === "returned") status.dataset.tone = "warning";
        info.append(status);
        if (state.view === "incoming" && Number.isInteger(image.object_count) && !image.negative) info.append(node("small", image.object_count ? count(image.object_count, "box", "boxes") : "No labels"));
        if (image.needs_decision) {
            const reason = image.review_required ? decisionReasons[image.review_required] || image.review_required.replaceAll("_", " ") : "imported labels differ from the saved ones";
            const decision = node("small", `Needs a decision: ${reason}`); decision.dataset.tone = "warning"; info.append(decision);
        }
        if (image.tranche_name) info.append(node("small", `Tranche: ${image.tranche_name}`));
        link.append(photo, info); card.append(link);
        if (state.session.user.role === "admin") {
            const history = node("button", "History", "card-history"); history.type = "button"; history.setAttribute("aria-label", `History of ${image.filename}`);
            history.onclick = () => window.TatorHostedHistory?.open(image.id, image.filename); card.append(history);
        }
        if (state.session.user.role === "admin" && selectable(image)) {
            const check = node("input"); check.type = "checkbox"; check.setAttribute("aria-label", `Select ${image.filename}`);
            check.dataset.image = image.id; check.checked = state.selected.has(image.id);
            check.onclick = event => { const at = state.images.findIndex(row => row.id === image.id), previous = state.images.findIndex(row => row.id === state.selectionAnchor); const rows = event.shiftKey && previous >= 0 ? state.images.slice(Math.min(at, previous), Math.max(at, previous) + 1) : [image]; for (const row of rows.filter(selectable)) { if (check.checked) state.selected.set(row.id, row); else state.selected.delete(row.id); } state.selectionAnchor = image.id; syncChecks(); }; card.append(check);
        }
        return card;
    }
    function drawMatchCount() {
        if (state.total === null) { $("imageMatchCount").textContent = ""; return; }
        const shown = state.images.length < state.total ? ` · ${state.images.length.toLocaleString()} shown` : "";
        $("imageMatchCount").textContent = `${count(state.total, "matching image")}${shown} · Shift-click to select a range${state.view === "incoming" ? " · Only Not assigned images can be selected" : ""}`;
        const remaining = state.total - state.images.length;
        $("moreAll").hidden = !state.cursor || remaining <= 100;
        $("moreAll").textContent = `Load all ${remaining.toLocaleString()} remaining`;
    }
    async function loadImages(append = false, limit = 100) {
        const sequence = ++state.sequence;
        const id = $("datasetChoice").value, captured = {sequence, dataset: id, view: state.view}, filters = JSON.stringify(state.filters);
        if (!append) refreshCounts();
        syncDatasetActions();
        const grouped = ["tranches", "mine"].includes(state.view);
        $("imageGrid").hidden = grouped; $("moreRow").hidden = grouped; $("empty").hidden = grouped;
        if (grouped && !id) {
            clearSelection(); state.images = []; window.TatorHostedTranches?.hide(); $("empty").hidden = false;
            $("empty").textContent = state.session.user.role === "admin" ? "Start with New dataset: name it and list its classes. Then add images to Incoming." : "Nothing is assigned to you yet. When an administrator assigns you a tranche, it appears here; use Refresh to check.";
            return;
        }
        if (grouped) { clearSelection(); state.images = []; await window.TatorHostedTranches?.show(); return; }
        window.TatorHostedTranches?.hide();
        if (!append) { state.cursor = null; state.total = null; clearSelection(); state.images = []; $("imageGrid").replaceChildren(); }
        if (!id) { $("empty").hidden = false; $("empty").textContent = state.session.user.role === "admin" ? "Start with New dataset: name it and list its classes. Then add images to Incoming." : "No editable work is currently assigned to you."; $("more").hidden = $("moreAll").hidden = true; return; }
        const query = queueQuery(limit);
        if (append && state.cursor) query.set("cursor", state.cursor);
        const result = await api(`/queue?${query}`);
        if (!currentImages(captured) || filters !== JSON.stringify(state.filters)) return;
        state.images.push(...result.images);
        const fragment = document.createDocumentFragment();
        for (const image of result.images) fragment.append(drawCard(image));
        $("imageGrid").append(fragment);
        $("empty").hidden = state.images.length !== 0;
        const filtered = state.filters.q || (state.view === "incoming" && state.filters.incoming !== "incoming") || (state.view === "submitted" && (state.filters.tranche || state.filters.annotator));
        $("empty").textContent = filtered ? "No images match these filters." : state.view === "submitted" ? "No images are waiting for review." : state.view === "confirmed" ? "No confirmed images yet. Add images in Incoming and approve them to build this dataset." : "No images in Incoming. Use Add images to upload new ones.";
        state.cursor = result.next_cursor; state.total = result.total; $("more").hidden = !state.cursor;
        drawMatchCount();
    }
    async function loadAll() {
        const remaining = state.total - state.images.length;
        if (remaining > 2000 && !window.confirm(`Load all ${remaining.toLocaleString()} remaining images? Very long lists can make this page slow; a search or filter keeps it quick.`)) return;
        const captured = {sequence: state.sequence, dataset: $("datasetChoice").value, view: state.view};
        while (state.cursor) {
            const before = state.sequence; await loadImages(true, 500);
            if (state.sequence !== before + 1 || captured.dataset !== $("datasetChoice").value || captured.view !== state.view) return;
        }
    }
    const viewCopy = { confirmed: ["Confirmed images", "Approved images used by analysis, training and dataset exports. New annotation work starts in Incoming."], incoming: ["Incoming", "New images wait here until they are approved. Group them into tranches for annotators, or approve labelled images yourself."], submitted: ["Review", "Images waiting for review. Return them with feedback, or approve them into the confirmed dataset."], tranches: ["Tranches", "Prepare a named group of Incoming images, assign it to one annotator, and review finished images as they arrive."], mine: ["My tranches", "Open a tranche to continue your work. Submit each image when it is ready; the remaining images stay with you."] };
    async function view(value, {route = null, tranche = null} = {}) {
        state.view = value;
        if (route) writeRoute(route, tranche);
        const captured = invalidateImages();
        const copy = state.session.user.role === "admin" && !state.datasets.length ? ["Create your first dataset", "A dataset holds its images, classes and annotations. Create one, then add images to its Incoming view."] : viewCopy[value];
        $("workspaceTitle").textContent = copy[0]; $("workspaceExplanation").textContent = copy[1];
        for (const button of $("views").children) button.setAttribute("aria-current", button.dataset.view === value ? "page" : "false");
        const isAdmin = state.session.user.role === "admin";
        $("import").hidden = !isAdmin || value !== "incoming";
        $("workspace").dataset.view = value;
        drawTitle(); drawImportReport();
        window.dispatchEvent(new Event("tator:hosted-view"));
        await prepareFilters(); if (!currentImages(captured)) return; await loadImages();
        if (state.session?.user.role === "annotator") await loadSubmissions();
    }
    async function switchDataset(id, route = "push") {
        if (!state.datasets.some(d => d.id === id)) throw new Error("Dataset unavailable for this account.");
        $("datasetChoice").value = id;
        resetFilters(); state.detail = null;
        window.dispatchEvent(new Event("tator:hosted-dataset"));
        await view(state.view, {route});
    }
    async function workspace() {
        window.dispatchEvent(new Event("tator:hosted-session"));
        const sequence = ++state.workspaceSequence;
        $("loginPanel").hidden = true; $("invitePanel").hidden = true; $("workspace").hidden = false; $("logout").hidden = false;
        $("accountName").textContent = `${state.session.user.username} · ${state.session.user.role === "admin" ? "Administrator" : "Annotator"}`; $("logout").title = `Signed in as ${$("accountName").textContent}`;
        const isAdmin = state.session.user.role === "admin";
        $("submissionPanel").hidden = isAdmin; $("classes").textContent = isAdmin ? "Classes" : "Classes & requests";
        $("createDataset").hidden = !isAdmin; $("people").hidden = !isAdmin; $("views").replaceChildren(); $("viewCounts").replaceChildren();
        for (const [key, label] of isAdmin ? [["confirmed", "Confirmed"], ["incoming", "Incoming"], ["tranches", "Tranches"], ["submitted", "Review"]] : [["mine", "My tranches"]]) {
            const button = node("button", label); button.type = "button"; button.dataset.view = key; button.id = `view-${key}`;
            if (key !== "tranches" && key !== "mine") {
                // The count is a description, so the tab's accessible name stays its label.
                const badge = node("span", "", "view-count"); badge.setAttribute("aria-hidden", "true");
                const description = node("span"); description.id = `${button.id}Count`; button.setAttribute("aria-describedby", description.id);
                button.append(badge); $("viewCounts").append(description);
            }
            button.onclick = () => action(button, () => navigate(key)); $("views").append(button);
        }
        const result = await api("/datasets");
        if (sequence !== state.workspaceSequence) return;
        state.datasets = result.datasets;
        if (!isAdmin) {
            // Finished tranches remain discoverable without granting access to
            // dataset classes, guidance, pixels or other people's annotations.
            let cursor = null;
            do {
                const params = new URLSearchParams({limit: "100"}); if (cursor) params.set("cursor", cursor);
                const personal = await api(`/tranches?${params}`);
                if (sequence !== state.workspaceSequence) return;
                for (const tranche of personal.tranches) if (!state.datasets.some(d => d.id === tranche.dataset_id)) state.datasets.push({id: tranche.dataset_id, name: tranche.dataset_name, metadata_only: true});
                cursor = personal.next_cursor;
            } while (cursor);
        }
        const route = new URLSearchParams(location.search), before = $("datasetChoice").value;
        const previous = route.get("dataset") || before; $("datasetChoice").replaceChildren();
        for (const dataset of state.datasets) { const option = node("option", dataset.name); option.value = dataset.id; $("datasetChoice").append(option); }
        if (state.datasets.some((row) => row.id === previous)) $("datasetChoice").value = previous;
        if (before && $("datasetChoice").value !== before) window.dispatchEvent(new Event("tator:hosted-dataset"));
        if (isAdmin) await refreshUsers();
        if (sequence !== state.workspaceSequence) return;
        syncDatasetActions();
        await applyRoute(route);
        if (sequence === state.workspaceSequence) window.dispatchEvent(new Event("tator:hosted-workspace"));
    }
    function personLabel(user) { return user?.display_name ? `${user.display_name} (${user.username})` : user?.username || ""; }
    function passwordLink(token, user, purpose = "invite") {
        const label = personLabel(user);
        $("invitationLink").value = `${location.origin}/#${new URLSearchParams({invite: token, user: user.username})}`;
        $("invitationExplanation").textContent = purpose === "invite"
            ? `Tator does not send email. Send this private link and the username “${user.username}” to ${label} yourself. They open it, choose a password and sign in with that username. It works once and expires after 24 hours.`
            : `Tator does not send email. Send this private link to ${label} yourself; their username is “${user.username}”. It lets them choose a new password, works once and expires after one hour. They have been signed out.`;
        $("invitation").hidden = false; $("invitation").scrollIntoView({ block: "nearest" });
    }
    function accountStatus(user) {
        const when = (seconds) => new Date(seconds * 1000).toLocaleString(), parts = [];
        if (user.disabled) parts.push("Access disabled");
        else if (user.password_state === "pending") parts.push(user.link && !user.link.expired ? `Invitation link created, not used yet · expires ${when(user.link.expires_at)}` : "No password yet · create a new invitation link or set a password");
        else { parts.push("Can sign in"); if (user.link && !user.link.expired) parts.push(`reset link open until ${when(user.link.expires_at)}`); }
        if (user.self) parts.push("You");
        if (user.last_administrator) parts.push("Only active administrator");
        if (user.open_work) parts.push(`${count(user.open_work, "open image")}`);
        return parts.join(" · ");
    }
    function drawUsers() {
        $("userList").replaceChildren();
        const labelled = (text, input) => { const label = node("label", text); label.append(input); return label; };
        for (const user of state.users) {
            const label = personLabel(user), path = `/users/${encodeURIComponent(user.id)}`;
            const form = node("form", undefined, "hosted-user"), heading = node("div", undefined, "hosted-user-heading");
            heading.append(node("strong", user.display_name || user.username), node("small", `@${user.username} · ${user.role === "admin" ? "Administrator" : "Annotator"}`));
            const status = node("p", accountStatus(user), "hosted-user-status");
            const fields = node("div", undefined, "hosted-inline-fields");
            const name = node("input"); name.value = user.display_name || ""; name.maxLength = 100; name.placeholder = user.username; name.setAttribute("aria-label", `Name shown for ${user.username}`);
            const role = node("select"); role.setAttribute("aria-label", "Role");
            for (const [value, text] of [["annotator", "Annotator"], ["admin", "Administrator"]]) { const option = node("option", text); option.value = value; role.append(option); }
            role.value = user.role;
            const accessLabel = node("label", undefined, "hosted-checkbox"), disabled = node("input"); disabled.type = "checkbox"; disabled.checked = Boolean(user.disabled);
            accessLabel.append(disabled, document.createTextNode("Disable access"));
            fields.append(labelled("Name shown in Tator", name), labelled("Role", role), accessLabel);
            const protectedAccount = user.self || user.last_administrator;
            if (protectedAccount) {
                role.disabled = disabled.disabled = true;
                fields.append(node("small", user.self ? "You cannot change your own role or access." : "Add another administrator before changing this one."));
            }
            const actions = node("div", undefined, "hosted-actions"), save = node("button", "Save changes");
            save.type = "submit"; save.disabled = true;
            const changed = () => { save.disabled = name.value.trim() === (user.display_name || "") && role.value === user.role && disabled.checked === Boolean(user.disabled); };
            name.oninput = changed; role.onchange = changed; disabled.onchange = changed;
            form.onsubmit = (event) => {
                event.preventDefault();
                const nextName = name.value.trim(), nextRole = role.value, nextDisabled = disabled.checked;
                const accessChanged = nextRole !== user.role || nextDisabled !== Boolean(user.disabled);
                if (accessChanged && !window.confirm(`Change ${label}'s ${nextDisabled !== Boolean(user.disabled) ? "access" : "role"}? ${nextDisabled ? "They lose access immediately." : "They will be signed out."} Saved work and assignments stay as they are.`)) return;
                action(save, async () => {
                    try {
                        // Access first: it can be refused (changed account, last administrator).
                        if (accessChanged) await api(path, { method: "PATCH", body: { role: nextRole, disabled: nextDisabled, expected_role: user.role, expected_disabled: Boolean(user.disabled) } });
                        if (nextName !== (user.display_name || "")) await api(`${path}/profile`, { method: "POST", body: { display_name: nextName || null } });
                    } catch (error) { await refreshUsers().catch(() => {}); throw error; }
                    await afterCommit(`Saved ${nextName || user.username}.`, refreshUsers);
                });
            };
            const passwordButton = node("button", user.self ? "Change my password" : "Set password"); passwordButton.type = "button"; passwordButton.disabled = Boolean(user.disabled);
            const passwordFields = node("div", undefined, "hosted-inline-fields hosted-password-fields"); passwordFields.hidden = true;
            const first = node("input"), second = node("input");
            for (const input of [first, second]) { input.type = "password"; input.minLength = 12; input.autocomplete = "new-password"; }
            const savePassword = node("button", "Save password"); savePassword.type = "button"; savePassword.className = "primary";
            passwordFields.append(labelled("New password, at least 12 characters", first), labelled("Repeat password", second), savePassword);
            passwordButton.onclick = () => { passwordFields.hidden = !passwordFields.hidden; if (!passwordFields.hidden) first.focus(); };
            savePassword.onclick = () => {
                if (first.value.length < 12) { notice("Use a password of at least 12 characters.", true); return; }
                if (first.value !== second.value) { notice("The passwords do not match.", true); return; }
                if (user.self && !window.confirm("Change your own password? You will be signed out everywhere, and any administrator compute you started will stop.")) return;
                action(savePassword, async () => {
                    await api(`${path}/password`, { method: "POST", body: { password: first.value } });
                    first.value = second.value = "";
                    if (user.self) { $("peopleDialog").close(); loginView(); notice("Your password changed. Sign in with the new password."); return; }
                    await afterCommit(`Password set for ${label}. Give it to them privately with their username “${user.username}”; any open session was signed out.`, refreshUsers);
                });
            };
            const linkButton = node("button", user.password_state === "pending" ? "New invitation link" : "Password reset link"); linkButton.type = "button"; linkButton.disabled = Boolean(user.disabled);
            linkButton.onclick = () => {
                const invite = user.password_state === "pending";
                if (!window.confirm(invite ? `Create a new invitation link for ${label}? Any earlier link stops working.` : `Create a password reset link for ${label}? They will be signed out and earlier links stop working.`)) return;
                action(linkButton, async () => {
                    const result = await api(`${path}/reset`, { method: "POST" });
                    passwordLink(result.reset_token, user, result.purpose || (invite ? "invite" : "reset"));
                    await afterCommit(invite ? `New invitation link created for ${label}. Copy it from People and send it with the username yourself.` : `Password reset link created for ${label}. Copy it from People and send it yourself.`, refreshUsers);
                });
            };
            actions.append(save, passwordButton);
            if (!user.self) actions.append(linkButton);
            if (user.link && !user.link.expired) {
                const cancel = node("button", user.link.purpose === "invite" ? "Cancel invitation" : "Cancel reset link"); cancel.type = "button";
                cancel.onclick = () => action(cancel, async () => { await api(`${path}/links/cancel`, { method: "POST" }); await afterCommit(`The open link for ${label} no longer works.`, refreshUsers); });
                actions.append(cancel);
            }
            if (!protectedAccount) {
                const remove = node("button", "Remove", "danger"); remove.type = "button"; remove.disabled = Boolean(user.open_work);
                remove.title = user.open_work ? "Review, recall or move this person's open images before removing them." : "Remove this account from the list.";
                remove.onclick = () => {
                    if (!window.confirm(`Remove ${label}? They can no longer sign in and disappear from this list. Their past work and review history stay attributed to them.`)) return;
                    action(remove, async () => { await api(`${path}/remove`, { method: "POST" }); await afterCommit(`${label} was removed.`, refreshUsers); });
                };
                actions.append(remove);
            }
            form.append(heading, status, fields, actions, passwordFields); $("userList").append(form);
        }
    }
    async function openReturned(submission) {
        // The server confirms the image is still yours and derives its tranche queue.
        const scope = await api(`/images/${encodeURIComponent(submission.image_id)}/scope`).catch(error => {
            if (!error.stale && [403, 404, 409].includes(error.status)) throw new Error(`${submission.filename} is no longer assigned to you. Refresh to see your current work.`);
            throw error;
        });
        const route = new URLSearchParams({image: submission.image_id, queue: scope.scope.queue, dataset: scope.scope.dataset_id, view: "mine"});
        if (scope.scope.tranche_id) route.set("tranche", scope.scope.tranche_id);
        location.assign(`/annotate?${route}`);
    }
    async function loadSubmissions(append = false) {
        if (state.session?.user.role !== "annotator") return;
        const sequence = ++state.submissionSequence, query = new URLSearchParams({ limit: "100", offset: String(append ? state.submissions.length : 0) });
        if (!append) { state.submissions = []; state.submissionImages = new Set(); $("submissionList").replaceChildren(); }
        const result = await api(`/submissions?${query}`);
        if (sequence !== state.submissionSequence) return;
        state.submissions.push(...result.submissions);
        for (const submission of result.submissions) {
            const row = node("article", undefined, "hosted-submission"), heading = node("div", undefined, "hosted-submission-heading");
            const label = { awaiting_review: "Waiting for review", returned: "Returned for changes", approved: "Confirmed", archived: "Archived by administrator" }[submission.status] || submission.status;
            const status = node("span", label, "hosted-status"); status.dataset.state = submission.status;
            if (submission.status === "returned") status.dataset.tone = "warning";
            heading.append(node("strong", submission.filename), status);
            row.append(heading, node("small", `${submission.dataset_name || "Dataset"} · Submitted ${new Date(submission.created_at * 1000).toLocaleString()}`));
            if (submission.feedback) row.append(node("p", submission.feedback));
            // Newest first: only an image's latest submission can still be returned work.
            const latest = !state.submissionImages.has(submission.image_id); state.submissionImages.add(submission.image_id);
            if (submission.status === "returned" && latest) {
                const link = node("a", "Open returned work"); link.href = `/annotate?${new URLSearchParams({image: submission.image_id, dataset: submission.dataset_id, view: "mine"})}`;
                link.onclick = event => { event.preventDefault(); action(link, () => openReturned(submission)); };
                row.append(link);
            }
            $("submissionList").append(row);
        }
        if (!state.submissions.length) $("submissionList").append(node("p", "You have no submitted images yet."));
        $("moreSubmissions").hidden = result.submissions.length < 100;
    }
    function drawGuidance(dataset, isAdmin) {
        const section = $("classGuidance"), body = $("classGuidanceBody");
        body.replaceChildren(); section.hidden = true;
        if (isAdmin) return;
        const context = String(dataset.settings?.context || "").trim(), glossary = String(dataset.settings?.labelmap_glossary || "").trim();
        if (!context && !glossary) return;
        if (context) body.append(node("p", context, "hosted-guidance-context"));
        let entries = null;
        try {
            const value = JSON.parse(glossary);
            if (value && typeof value === "object" && !Array.isArray(value)) entries = Object.entries(value).map(([name, terms]) => [name, Array.isArray(terms) ? terms.join(", ") : String(terms)]);
        } catch (_) {
            const lines = glossary.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
            if (lines.length && lines.every(line => line.includes(":"))) entries = lines.map(line => [line.slice(0, line.indexOf(":")).trim(), line.slice(line.indexOf(":") + 1).trim()]);
        }
        if (entries?.length) { const list = node("dl", undefined, "hosted-glossary"); for (const [name, meaning] of entries) list.append(node("dt", name), node("dd", meaning)); body.append(list); }
        else if (glossary) body.append(node("p", glossary, "hosted-guidance-context"));
        section.hidden = false;
    }
    async function loadClasses() {
        const id = state.classDataset?.id || $("datasetChoice").value, sequence = ++state.classSequence;
        if (!id) return;
        const [dataset, requests] = await Promise.all([api(`/datasets/${encodeURIComponent(id)}`), api(`/class-requests?dataset_id=${encodeURIComponent(id)}`)]);
        if (sequence !== state.classSequence || !$("classesDialog").open) return;
        state.classDataset = dataset;
        const isAdmin = state.session.user.role === "admin";
        $("classesTitle").textContent = `Classes · ${dataset.name}`;
        $("classesExplanation").textContent = isAdmin ? "Class names are shared by this dataset. Renaming preserves each class's identity and existing annotations. Describe what each class means in Dataset guidance; annotators see it here." : "Use the dataset's existing classes. Request a new class here when you need one; an administrator reviews it.";
        $("addClassForm").hidden = !isAdmin; $("requestClassForm").hidden = isAdmin;
        $("classList").replaceChildren(); $("classRequestList").replaceChildren();
        drawGuidance(dataset, isAdmin);
        for (const item of dataset.classes) {
            const row = node(isAdmin ? "form" : "div", undefined, "hosted-class");
            if (isAdmin) {
                const label = node("label", "Class name"), input = node("input"); input.value = item.name; input.required = true; input.maxLength = 200; label.append(input);
                const rename = node("button", "Rename"); rename.type = "submit"; rename.disabled = true;
                input.oninput = () => { rename.disabled = !input.value.trim() || input.value.trim() === item.name; };
                row.append(label, rename);
                row.onsubmit = (event) => { event.preventDefault(); const name = input.value.trim(); action(rename, () => classCommand(`/datasets/${encodeURIComponent(dataset.id)}/classes/${encodeURIComponent(item.id)}`, { name, expected_class_version: dataset.class_version }, "Class renamed.", "PATCH")); };
            } else row.append(node("span", item.name));
            $("classList").append(row);
        }
        if (!dataset.classes.length) $("classList").append(node("p", "No classes have been added yet."));
        for (const request of requests.requests) {
            const row = node("article", undefined, "hosted-class-request"), author = state.users.find((user) => user.id === request.actor_id);
            row.append(node("strong", request.name), node("small", `${request.state === "pending" ? "Waiting for review" : request.state === "approved" ? "Approved" : "Declined"}${isAdmin && author ? ` · ${author.username}` : ""}`), node("p", request.reason));
            if (request.feedback) row.append(node("p", `Feedback: ${request.feedback}`));
            if (isAdmin && request.state === "pending") {
                const form = node("form"), label = node("label", "Feedback (optional)"), feedback = node("textarea"); feedback.rows = 2; feedback.maxLength = 5000; label.append(feedback);
                const actions = node("div", undefined, "hosted-actions"), approve = node("button", "Approve class", "primary"), decline = node("button", "Decline");
                approve.type = decline.type = "button";
                const resolve = (button, approved) => action(button, () => classCommand(`/class-requests/${encodeURIComponent(request.id)}/resolve`, { approved, feedback: feedback.value.trim(), expected_class_version: dataset.class_version }, approved ? "Class request approved." : "Class request declined."));
                approve.onclick = () => resolve(approve, true); decline.onclick = () => resolve(decline, false);
                actions.append(approve, decline); form.append(label, actions); row.append(form);
            }
            $("classRequestList").append(row);
        }
        if (!requests.requests.length) $("classRequestList").append(node("p", "No class requests for this dataset."));
    }
    async function classCommand(path, body, message, method = "POST") {
        const id = state.classDataset?.id;
        await api(path, { method, body: { ...body, operation_id: op() } });
        await afterCommit(message, async () => { if ($("classesDialog").open && state.classDataset?.id === id) await loadClasses(); });
    }

    // ---- Explicit bulk approval: one confirmation, chunked requests, one receipt per image.
    const skipReasons = {editor_lease_conflict: "open for editing", annotation_version_changed: "changed since you selected it", ownership_changed: "changed since you selected it", image_not_assignable: "now in a tranche or assigned", image_needs_decision: "needs a decision", image_has_no_labels: "has no labels", image_has_labels: "has labels", image_not_approvable: "no longer waiting for review", annotation_not_complete: "not marked as checked", empty_annotation_requires_negative_review: "has no boxes and is not marked as empty", image_not_found: "no longer available", image_dataset_mismatch: "in another dataset"};
    async function bulkApprove(kind, rows) {
        const dataset = $("datasetChoice").value, authority = state.session, editor = op(), names = new Map(rows.map(row => [row.id, row.filename]));
        const path = `/datasets/${encodeURIComponent(dataset)}/${kind === "review" ? "approve-images" : "confirm-incoming"}`;
        const items = rows.map(image => ({image_id: image.id, expected_version: image.current_version, generation: image.generation, operation_id: op()}));
        let approved = 0; const skipped = [];
        const summary = () => {
            const groups = new Map();
            for (const row of skipped) { const reason = skipReasons[row.error] || errorText(row.error, row.detail).replace(/\.$/, ""); if (!groups.has(reason)) groups.set(reason, []); groups.get(reason).push(names.get(row.image_id) || row.image_id); }
            const detail = [...groups].map(([reason, files]) => `${files.length.toLocaleString()} ${reason} (${files.slice(0, 3).join(", ")}${files.length > 3 ? `, and ${(files.length - 3).toLocaleString()} more` : ""})`).join("; ");
            return `${count(approved, "image")} ${kind === "background" ? "confirmed as background" : "approved"}.${skipped.length ? ` ${skipped.length.toLocaleString()} skipped and left unchanged: ${detail}.` : ""}`;
        };
        try {
            for (let start = 0; start < items.length; start += BULK_CHUNK) {
                if (authority !== state.session || dataset !== $("datasetChoice").value) throw new Error("The account or dataset changed, so the rest was not sent.");
                if (items.length > BULK_CHUNK) notice(`Approving ${(start + 1).toLocaleString()}–${Math.min(start + BULK_CHUNK, items.length).toLocaleString()} of ${items.length.toLocaleString()}…`);
                const body = {operation_id: op(), editor_id: editor, images: items.slice(start, start + BULK_CHUNK), ...(kind === "review" ? {} : {mode: kind})};
                const result = await api(path, {method: "POST", body});
                for (const row of result.images) { if (row.status === "approved") approved++; else skipped.push(row); }
            }
        } catch (error) {
            if (error.stale) throw error;
            notice(`${summary()} Stopped before the rest: ${error.message} Refresh, then select the remaining images to continue.`, true);
            await loadImages().catch(() => {}); return;
        }
        notice(summary(), Boolean(skipped.length));
        await loadImages();
    }
    $("approveIncoming").onclick = () => action($("approveIncoming"), async () => {
        const rows = [...state.selected.values()].filter(row => row.object_count > 0 && !row.needs_decision);
        if (!rows.length) return;
        if (!window.confirm(`Approve ${count(rows.length, "selected Incoming image")} with ${rows.length === 1 ? "its" : "their"} current labels into the confirmed dataset of ${datasetName()}? Only continue after checking the annotations. Images that changed, are open for editing or need a decision are skipped.`)) return;
        await bulkApprove("labelled", rows);
    });
    $("confirmBackground").onclick = () => action($("confirmBackground"), async () => {
        const rows = [...state.selected.values()].filter(row => !row.object_count && !row.needs_decision);
        if (!rows.length) return;
        if (!window.confirm(`Confirm ${count(rows.length, "selected image")} without labels as background — no target objects — and approve ${rows.length === 1 ? "it" : "them"} into the confirmed dataset of ${datasetName()}? Only continue after checking that they contain none of this dataset's classes.`)) return;
        await bulkApprove("background", rows);
    });
    $("approveSelected").onclick = () => action($("approveSelected"), async () => {
        const selected = Array.from(state.selected.values());
        if (!confirm(`Approve ${count(selected.length, "submitted image")} into the confirmed dataset? Only continue after checking their annotations.`)) return;
        await bulkApprove("review", selected);
    });

    $("loginForm").onsubmit = (event) => { event.preventDefault(); const form = new FormData(event.currentTarget); action(event.submitter, async () => { state.session = await api("/login", { method: "POST", body: { username: form.get("username"), password: form.get("password") } }); event.target.reset(); announceSession(); notice(""); const next = nextPage(); if (next) { location.replace(next); return; } await workspace(); }); };
    $("logout").onclick = () => action($("logout"), async () => { await api("/logout", { method: "POST" }); announceSession(); state.selected.clear(); state.images = []; $("imageGrid").replaceChildren(); loginView(); });
    $("refresh").onclick = () => action($("refresh"), workspace);
    $("datasetChoice").onchange = () => { notice(""); switchDataset($("datasetChoice").value).catch((error) => { if (!error.stale) notice(error.message, true); }); };
    $("more").onclick = () => action($("more"), () => loadImages(true));
    $("moreAll").onclick = () => action($("moreAll"), loadAll);
    $("clearSelection").onclick = clearSelection; $("assignee").onchange = selection;
    $("createDataset").onclick = () => { notice(""); $("datasetDialog").showModal(); };
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    $("datasetForm").onsubmit = (event) => {
        event.preventDefault(); const form = new FormData(event.currentTarget);
        const name = String(form.get("name")).trim(), classes = String(form.get("classes")).split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
        // Say which line repeats before asking the server, which compares names without capitals.
        const seen = new Set(), repeated = classes.find(item => { const key = item.toLocaleLowerCase(); if (seen.has(key)) return true; seen.add(key); return false; });
        if (repeated) { notice(`“${repeated}” is listed twice; class names are compared without capitals. List each class once.`, true); return; }
        if (state.datasets.some(row => row.name.trim().toLocaleLowerCase() === name.toLocaleLowerCase()) && !window.confirm(`A dataset named “${name}” already exists. Create another one with the same name?`)) return;
        action(event.submitter, async () => {
            const created = await api("/datasets", { method: "POST", body: { name, classes, operation_id: op() } });
            $("datasetDialog").close(); event.target.reset();
            // The new dataset opens in Incoming, where its images are added.
            history.pushState(null, "", `/?${new URLSearchParams({dataset: created.id, view: "incoming"})}`);
            state.view = "incoming"; resetFilters(); await workspace();
            notice(`Dataset created: ${created.name}. Add images to Incoming to start.`);
        });
    };
    $("assign").onclick = () => window.TatorHostedTranches?.create([...state.selected.values()]);
    $("people").onclick = () => { $("invitation").hidden = true; $("invitationLink").value = ""; $("peopleDialog").showModal(); notice(""); action($("people"), refreshUsers); };
    $("refreshPeople").onclick = () => action($("refreshPeople"), refreshUsers);
    const syncPersonAccess = () => { const form = $("personForm"), password = form.elements.access.value === "password"; $("personPasswordFields").hidden = !password; for (const name of ["password", "repeat"]) form.elements[name].required = password; };
    for (const radio of $("personForm").querySelectorAll('input[name="access"]')) radio.onchange = syncPersonAccess;
    syncPersonAccess();
    $("personForm").onsubmit = (event) => {
        event.preventDefault();
        const form = new FormData(event.currentTarget), access = form.get("access");
        if (access === "password" && form.get("password") !== form.get("repeat")) { notice("The passwords do not match.", true); return; }
        action(event.submitter, async () => {
            const body = { username: form.get("username"), role: form.get("role"), display_name: String(form.get("display_name") || "").trim() || null };
            if (access === "password") body.password = form.get("password");
            const person = await api("/users", { method: "POST", body });
            event.target.reset(); syncPersonAccess();
            if (person.invite_token) passwordLink(person.invite_token, person, "invite");
            await afterCommit(person.invite_token ? `Invitation link created for ${personLabel(person)}. Copy it from People and send it with the username “${person.username}” yourself.` : `${personLabel(person)} can now sign in as “${person.username}” with the password you set.`, refreshUsers);
        });
    };
    $("copyInvitation").onclick = async () => {
        try { await navigator.clipboard.writeText($("invitationLink").value); notice("Link copied."); }
        catch (_) { $("invitationLink").select(); notice("Select the link and copy it with your keyboard."); }
    };
    $("classes").onclick = () => { state.classDataset = null; $("classList").replaceChildren(); $("classRequestList").replaceChildren(); $("classGuidance").hidden = true; $("classesDialog").showModal(); notice(""); action($("classes"), loadClasses); };
    $("classesDialog").addEventListener("close", () => { state.classSequence++; state.classDataset = null; });
    $("refreshClasses").onclick = () => action($("refreshClasses"), loadClasses);
    $("addClassForm").onsubmit = (event) => {
        event.preventDefault(); const dataset = state.classDataset, form = new FormData(event.currentTarget); if (!dataset) return;
        action(event.submitter, async () => { await classCommand(`/datasets/${encodeURIComponent(dataset.id)}/classes`, { name: String(form.get("name")).trim(), expected_class_version: dataset.class_version }, "Class added."); event.target.reset(); });
    };
    $("requestClassForm").onsubmit = (event) => {
        event.preventDefault(); const dataset = state.classDataset, form = new FormData(event.currentTarget); if (!dataset) return;
        action(event.submitter, async () => { await classCommand(`/datasets/${encodeURIComponent(dataset.id)}/class-requests`, { name: String(form.get("name")).trim(), reason: String(form.get("reason")).trim() }, "Class request sent."); event.target.reset(); });
    };
    $("moreSubmissions").onclick = () => action($("moreSubmissions"), () => loadSubmissions(true));
    $("inviteForm").onsubmit = (event) => {
        event.preventDefault(); const form = new FormData(event.currentTarget);
        action(event.submitter, async () => {
            if (form.get("password") !== form.get("repeat")) throw new Error("The passwords do not match.");
            const link = new URLSearchParams(location.hash.slice(1)), username = $("inviteUsername").value;
            await api("/redeem", { method: "POST", body: { token: link.get("invite"), password: form.get("password") } });
            history.replaceState(null, "", "/"); event.target.reset(); $("invitePanel").hidden = true; loginView();
            const login = $("loginForm").elements;
            if (username) { login.username.value = username; login.password.focus(); } else login.username.focus();
            notice(username ? `Password saved. Sign in as ${username} with your new password.` : "Password saved. Sign in with the username your administrator sent you and your new password.");
        });
    };
    window.addEventListener("hashchange", () => { if (!invitationView() && !$("invitePanel").hidden) { $("invitePanel").hidden = true; loginView(); } });
    (async () => {
        if (invitationView()) return;
        try { state.session = await api("/session"); const next = nextPage(); if (next) { location.replace(next); return; } await workspace(); }
        catch (error) { if (!error.stale) { loginView(); if (error.status !== 401) notice(error.message, true); else if (nextPage()) notice("Sign in to open the page you asked for."); } }
    })();
    const applyFilters = () => {
        state.filters = {q: $("imageSearch").value.trim(), incoming: $("incomingFilter").value, tranche: $("reviewTranche").value, annotator: $("reviewAnnotator").value};
        writeRoute("replace");
        return loadImages();
    };
    $("imageFilters").onsubmit = event => { event.preventDefault(); action(event.submitter, applyFilters); };
    for (const id of ["incomingFilter", "reviewTranche", "reviewAnnotator"]) $(id).addEventListener("change", () => action($(id), applyFilters));
    $("selectVisibleImages").onclick = event => action(event.currentTarget, () => { for (const image of state.images.filter(selectable)) state.selected.set(image.id, image); syncChecks(); });
    $("selectMatchingImages").onclick = event => action(event.currentTarget, selectMatching);
    window.TatorHostedDashboard = { api, action, notice, errorText, refresh: workspace, refreshCounts, dropRouteParams,
        selectDataset: id => id === $("datasetChoice").value ? Promise.resolve() : switchDataset(id, "replace"),
        showView: view, navigate, loadImages, clearSelection, get view() { return state.view; }, get users() { return state.users; }, personLabel,
        get session() { return state.session; }, get selectedImages() { return [...state.selected.values()]; }, get datasetId() { return $("datasetChoice").value; },
        get datasetName() { return datasetName(); } };
})();
