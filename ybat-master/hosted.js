(() => {
    "use strict";
    const $ = (id) => document.getElementById(id);
    const state = { session: null, sessionEpoch: 0, datasets: [], images: [], selected: new Map(), users: [], view: "confirmed", cursor: null, sequence: 0, workspaceSequence: 0, userSequence: 0, classSequence: 0, submissionSequence: 0, classDataset: null, submissions: [], selectionAnchor: null, filters: {q: "", incoming: "incoming", tranche: "", annotator: ""} };
    const op = () => crypto.randomUUID();
    const node = (tag, text, className) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; };
    function notice(message, error = false) {
        for (const element of [$("notice"), ...document.querySelectorAll("dialog[open] [data-dialog-notice]")]) { element.textContent = message; element.dataset.error = String(error); element.hidden = !message; }
    }
    const errorMessages = {tranche_revision_changed: "The tranche changed. Refresh it before applying this change.", tranche_capacity_exceeded: "A tranche can hold at most 10,000 unfinished images. Create another tranche for the remaining images.", image_left_queue: "That image left this queue. Reopen the tranche to continue.", tranche_has_pending_review: "Review the submitted images first, or move selected unfinished images to another tranche.", tranche_has_unfinished_members: "Finish or remove the remaining work before closing this tranche.", image_not_assignable: "An image is no longer available in Incoming. Refresh the selection before trying again.", dataset_revision_changed: "The dataset changed. Refresh the preview before applying this change.", split_group_conflict: "These images share a capture group. Preview the complete group before changing its split.", editor_lease_conflict: "An image is open for editing. Close its editing tab before changing this selection.", last_administrator: "Keep at least one active administrator.", username_exists: "That username is already used, possibly by a removed account. Choose a different username.", class_version_changed: "The class list changed. Refresh classes and review the current names before trying again.", class_request_already_reviewed: "This request has already been reviewed. Refresh classes to see the result.", class_name_exists: "That class name already exists.", user_changed: "This account changed. Refresh People before applying your change.", installation_changed: "The server changed. Sign in again before continuing." };
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
                    : errorMessages[data.error] || (typeof data.detail === "string" ? data.detail : data.error) || `Request failed (${response.status})`);
                error.status = response.status; error.code = data.error;
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
    function applyTheme(dark) { document.documentElement.dataset.theme = dark ? "dark" : "light"; $("theme").textContent = dark ? "Light mode" : "Dark mode"; localStorage.setItem("tator.hosted.theme", dark ? "dark" : "light"); }
    $("theme").onclick = () => applyTheme(document.documentElement.dataset.theme !== "dark");
    applyTheme(localStorage.getItem("tator.hosted.theme") === "dark" || (!localStorage.getItem("tator.hosted.theme") && matchMedia("(prefers-color-scheme: dark)").matches));
    function announceSession() { try { localStorage.setItem("tator.hosted.session-event", crypto.randomUUID()); } catch (_) {} }
    window.addEventListener("storage", event => { if (event.key === "tator.hosted.session-event" && state.session) { loginView(); notice("The signed-in account changed in another tab. Sign in or reload to continue."); } });
    function loginView() {
        state.session = null; state.sessionEpoch++; state.sequence++; state.workspaceSequence++; state.userSequence++; state.classSequence++; state.submissionSequence++;
        state.images = []; state.datasets = []; state.users = []; state.submissions = []; state.classDataset = null; state.selected.clear();
        for (const id of ["imageGrid", "userList", "classList", "classRequestList", "submissionList", "datasetChoice", "assignee"]) $(id).replaceChildren();
        document.querySelectorAll("dialog[open]").forEach((dialog) => dialog.close());
        $("invitationLink").value = ""; $("invitation").hidden = true;
        $("workspace").hidden = true; $("loginPanel").hidden = false; $("logout").hidden = true; $("accountName").textContent = "";
        window.dispatchEvent(new Event("tator:hosted-session"));
    }
    function invitationView() {
        if (!new URLSearchParams(location.hash.slice(1)).has("invite")) return false;
        loginView(); $("loginPanel").hidden = true; $("invitePanel").hidden = false; notice(""); return true;
    }
    function selection() {
        $("selectionBar").hidden = state.selected.size === 0;
        $("selectionCount").textContent = `${state.selected.size} image${state.selected.size === 1 ? "" : "s"} selected`;
        $("assign").hidden = state.view !== "incoming"; $("assigneeLabel").hidden = true; $("assign").textContent = "Create tranche from selection";
        $("approveSelected").hidden = state.view !== "submitted";
        $("assign").disabled = state.selected.size === 0;
        window.dispatchEvent(new Event("tator:hosted-selection"));
    }
    function clearSelection() { state.selected.clear(); state.selectionAnchor = null; document.querySelectorAll(".hosted-card input").forEach((input) => { input.checked = false; }); selection(); }
    function invalidateImages() {
        ++state.sequence; clearSelection(); state.images = []; state.cursor = null;
        $("imageGrid").replaceChildren(); $("imageMatchCount").textContent = ""; $("more").hidden = true;
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
        const authority = state.session, captured = {sequence: state.sequence, dataset: $("datasetChoice").value, view: state.view}, filters = JSON.stringify(state.filters), query = queueQuery(500);
        if (state.view === "incoming") query.set("queue", "unallocated");
        const result = await api(`/queue?${query}`);
        if (authority !== state.session || !currentImages(captured) || filters !== JSON.stringify(state.filters)) return;
        if (result.total > 500 || result.next_cursor) throw new Error(`${result.total} eligible images match. Narrow the search to at most 500 for one exact selection.`);
        if (!window.confirm(`Select all ${result.images.length} eligible matching images, including those not yet displayed?`)) return;
        if (authority !== state.session || !currentImages(captured) || filters !== JSON.stringify(state.filters)) return;
        state.selected = new Map(result.images.map(image => [image.id, image])); syncChecks();
    }
    async function prepareFilters() {
        const isAdmin = state.session?.user.role === "admin", flat = !["tranches", "mine"].includes(state.view);
        $("imageFilters").hidden = !isAdmin || !flat;
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
    function drawCard(image) {
        const card = node("article", undefined, "hosted-card");
        const link = node("a"), route = new URLSearchParams({image: image.id});
        // Incoming includes every intake state; let the server derive each
        // image's actual tranche scope instead of widening its canvas queue.
        if (state.view !== "incoming") { route.set("queue", state.view === "submitted" ? "review" : "confirmed"); if (state.view === "submitted" && state.filters.tranche) route.set("tranche", state.filters.tranche); }
        link.href = `/annotate?${route}`;
        const photo = node("img"); photo.src = `/api/hosted/images/${encodeURIComponent(image.id)}/media`; photo.alt = image.filename; photo.loading = "lazy";
        const info = node("div", undefined, "card-info"); info.append(node("strong", image.filename));
        const labels = { confirmed: "Confirmed", assigned: "Assigned", returned: "Changes requested", submitted: "Ready for review", unassigned: "Not assigned" };
        info.append(node("small", `${labels[image.state] || image.state}${image.negative ? " · No target objects" : ""}`));
        if (image.tranche_name) info.append(node("small", `Tranche: ${image.tranche_name}${image.assignee_name ? ` · ${image.assignee_name}` : ""}`));
        link.append(photo, info); card.append(link);
        if (state.session.user.role === "admin" && selectable(image)) {
            const check = node("input"); check.type = "checkbox"; check.setAttribute("aria-label", `Select ${image.filename}`);
            check.dataset.image = image.id; check.checked = state.selected.has(image.id);
            check.onclick = event => { const at = state.images.findIndex(row => row.id === image.id), previous = state.images.findIndex(row => row.id === state.selectionAnchor); const rows = event.shiftKey && previous >= 0 ? state.images.slice(Math.min(at, previous), Math.max(at, previous) + 1) : [image]; for (const row of rows.filter(selectable)) { if (check.checked) state.selected.set(row.id, row); else state.selected.delete(row.id); } state.selectionAnchor = image.id; syncChecks(); }; card.append(check);
        }
        return card;
    }
    async function loadImages(append = false) {
        const sequence = ++state.sequence;
        const id = $("datasetChoice").value, captured = {sequence, dataset: id, view: state.view}, filters = JSON.stringify(state.filters);
        $("classes").disabled = !id || Boolean(state.datasets.find(d => d.id === id)?.metadata_only);
        const grouped = ["tranches", "mine"].includes(state.view);
        $("imageGrid").hidden = grouped; $("more").hidden = grouped; $("empty").hidden = grouped;
        if (grouped) { clearSelection(); state.images = []; await window.TatorHostedTranches?.show(); return; }
        window.TatorHostedTranches?.hide();
        if (!append) state.cursor = null;
        if (!append) { clearSelection(); state.images = []; $("imageGrid").replaceChildren(); }
        if (!id) { $("empty").hidden = false; $("empty").textContent = state.session.user.role === "admin" ? "Create a dataset, then add images to Incoming." : "No editable work is currently assigned to you."; $("more").hidden = true; return; }
        const query = queueQuery();
        if (append && state.cursor) query.set("cursor", state.cursor);
        const result = await api(`/queue?${query}`);
        if (!currentImages(captured) || filters !== JSON.stringify(state.filters)) return;
        state.images.push(...result.images);
        for (const image of result.images) $("imageGrid").append(drawCard(image));
        $("empty").hidden = state.images.length !== 0;
        $("empty").textContent = state.view === "submitted" ? "No images are waiting for review." : state.view === "confirmed" ? "No confirmed images yet. Add and approve Incoming images to build this dataset." : "No images in this view.";
        state.cursor = result.next_cursor; $("more").hidden = !state.cursor;
        $("imageMatchCount").textContent = `${result.total} matching images · Shift-click to select a range${state.view === "incoming" ? " · Only unallocated images can be selected" : ""}`;
    }
    async function view(value) {
        state.view = value;
        const captured = invalidateImages();
        const copy = { confirmed: ["Confirmed images", "Approved images used by analysis, training and dataset exports. They are finished and not assigned to anyone; new annotation work starts in Incoming."], incoming: ["Incoming", "Prepare and assign new images. They join the dataset only after approval."], submitted: ["Review", "Check submitted work, return feedback, or approve it into the dataset."], tranches: ["Annotation tranches", "Prepare a named group of Incoming images, assign it to one annotator, and review finished images as they arrive."], mine: ["My tranches", "Open a tranche to continue your work. Submit each image when it is ready; the remaining images stay with you."] }[value];
        $("workspaceTitle").textContent = copy[0]; $("workspaceExplanation").textContent = copy[1];
        for (const button of $("views").children) button.setAttribute("aria-current", button.dataset.view === value ? "page" : "false");
        $("import").hidden = state.session.user.role !== "admin" || value !== "incoming";
        await prepareFilters(); if (!currentImages(captured)) return; await loadImages();
        if (state.session?.user.role === "annotator") await loadSubmissions();
    }
    async function workspace() {
        window.dispatchEvent(new Event("tator:hosted-session"));
        const sequence = ++state.workspaceSequence;
        $("loginPanel").hidden = true; $("invitePanel").hidden = true; $("workspace").hidden = false; $("logout").hidden = false;
        $("accountName").textContent = `${state.session.user.username} · ${state.session.user.role === "admin" ? "Administrator" : "Annotator"}`;
        const isAdmin = state.session.user.role === "admin";
        $("submissionPanel").hidden = isAdmin; $("classes").textContent = isAdmin ? "Classes" : "Classes & requests";
        $("createDataset").hidden = !isAdmin; $("people").hidden = !isAdmin; $("views").replaceChildren();
        for (const [key, label] of isAdmin ? [["confirmed", "Dataset"], ["incoming", "Incoming"], ["tranches", "Tranches"], ["submitted", "Review"]] : [["mine", "My tranches"]]) {
            const button = node("button", label); button.type = "button"; button.dataset.view = key; button.onclick = () => action(button, () => { history.replaceState(null, "", `/?${new URLSearchParams({dataset: $("datasetChoice").value, view: key})}`); return view(key); }); $("views").append(button);
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
        const previous = $("datasetChoice").value || new URLSearchParams(location.search).get("dataset"); $("datasetChoice").replaceChildren();
        for (const dataset of state.datasets) { const option = node("option", dataset.name); option.value = dataset.id; $("datasetChoice").append(option); }
        if (state.datasets.some((row) => row.id === previous)) $("datasetChoice").value = previous;
        if (isAdmin) await refreshUsers();
        if (sequence !== state.workspaceSequence) return;
        const route = new URLSearchParams(location.search);
        if (route.get("tranche") || route.get("view") === "tranches") state.view = "tranches";
        else if (["incoming", "submitted", "confirmed"].includes(route.get("view"))) state.view = route.get("view");
        await view(isAdmin ? (state.view === "mine" ? "confirmed" : state.view) : "mine");
        if (sequence === state.workspaceSequence) window.dispatchEvent(new Event("tator:hosted-workspace"));
    }
    function personLabel(user) { return user?.display_name ? `${user.display_name} (${user.username})` : user?.username || ""; }
    function passwordLink(token, label, purpose = "invite") {
        $("invitationLink").value = `${location.origin}/#invite=${encodeURIComponent(token)}`;
        $("invitationExplanation").textContent = purpose === "invite"
            ? `Send this private link to ${label} yourself. They open it, choose a password and sign in. It works once and expires after 24 hours.`
            : `Send this private link to ${label} yourself. It lets them choose a new password, works once and expires after one hour. They have been signed out.`;
        $("invitation").hidden = false; $("invitation").scrollIntoView({ block: "nearest" });
    }
    function accountStatus(user) {
        const when = (seconds) => new Date(seconds * 1000).toLocaleString(), parts = [];
        if (user.disabled) parts.push("Access disabled");
        else if (user.password_state === "pending") parts.push(user.link && !user.link.expired ? `Invitation sent, not used yet · expires ${when(user.link.expires_at)}` : "No password yet · create a new invitation link or set a password");
        else { parts.push("Can sign in"); if (user.link && !user.link.expired) parts.push(`reset link open until ${when(user.link.expires_at)}`); }
        if (user.self) parts.push("You");
        if (user.last_administrator) parts.push("Only active administrator");
        if (user.open_work) parts.push(`${user.open_work} open image${user.open_work === 1 ? "" : "s"}`);
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
                    await afterCommit(`Password set for ${label}. Give it to them privately; any open session was signed out.`, refreshUsers);
                });
            };
            const linkButton = node("button", user.password_state === "pending" ? "New invitation link" : "Password reset link"); linkButton.type = "button"; linkButton.disabled = Boolean(user.disabled);
            linkButton.onclick = () => {
                const invite = user.password_state === "pending";
                if (!window.confirm(invite ? `Create a new invitation link for ${label}? Any earlier link stops working.` : `Create a password reset link for ${label}? They will be signed out and earlier links stop working.`)) return;
                action(linkButton, async () => {
                    const result = await api(`${path}/reset`, { method: "POST" });
                    passwordLink(result.reset_token, label, result.purpose || (invite ? "invite" : "reset"));
                    await afterCommit(invite ? `New invitation link created for ${label}. Copy it from People.` : `Password reset link created for ${label}. Copy it from People.`, refreshUsers);
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
    async function loadSubmissions(append = false) {
        if (state.session?.user.role !== "annotator") return;
        const sequence = ++state.submissionSequence, query = new URLSearchParams({ limit: "100", offset: String(append ? state.submissions.length : 0) });
        if (!append) { state.submissions = []; $("submissionList").replaceChildren(); }
        const result = await api(`/submissions?${query}`);
        if (sequence !== state.submissionSequence) return;
        state.submissions.push(...result.submissions);
        for (const submission of result.submissions) {
            const row = node("article", undefined, "hosted-submission"), heading = node("div", undefined, "hosted-submission-heading");
            const status = { awaiting_review: "Awaiting review", returned: "Changes requested", approved: "Approved", archived: "Archived by administrator" }[submission.status] || submission.status;
            heading.append(node("strong", submission.filename), node("span", status, "hosted-status"));
            row.append(heading, node("small", `${submission.dataset_name || "Dataset"} · Submitted ${new Date(submission.created_at * 1000).toLocaleString()}`));
            if (submission.feedback) row.append(node("p", submission.feedback));
            if (submission.status === "returned" && state.images.some((image) => image.id === submission.image_id && image.state === "returned")) {
                const link = node("a", "Open returned work"); link.href = `/annotate?image=${encodeURIComponent(submission.image_id)}`; row.append(link);
            }
            $("submissionList").append(row);
        }
        if (!state.submissions.length) $("submissionList").append(node("p", "You have no submitted images yet."));
        $("moreSubmissions").hidden = result.submissions.length < 100;
    }
    async function loadClasses() {
        const id = state.classDataset?.id || $("datasetChoice").value, sequence = ++state.classSequence;
        if (!id) return;
        const [dataset, requests] = await Promise.all([api(`/datasets/${encodeURIComponent(id)}`), api(`/class-requests?dataset_id=${encodeURIComponent(id)}`)]);
        if (sequence !== state.classSequence || !$("classesDialog").open) return;
        state.classDataset = dataset;
        const isAdmin = state.session.user.role === "admin";
        $("classesTitle").textContent = `${dataset.name} · Classes`;
        $("classesExplanation").textContent = isAdmin ? "Class names are shared by this dataset. Renaming preserves each class's identity and existing annotations." : "Use the dataset's existing classes. Request a new class here when you need one; an administrator reviews it.";
        $("addClassForm").hidden = !isAdmin; $("requestClassForm").hidden = isAdmin;
        $("classList").replaceChildren(); $("classRequestList").replaceChildren();
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
            row.append(node("strong", request.name), node("small", `${request.state === "pending" ? "Awaiting review" : request.state === "approved" ? "Approved" : "Declined"}${isAdmin && author ? ` · ${author.username}` : ""}`), node("p", request.reason));
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
    $("loginForm").onsubmit = (event) => { event.preventDefault(); const form = new FormData(event.currentTarget); action(event.submitter, async () => { state.session = await api("/login", { method: "POST", body: { username: form.get("username"), password: form.get("password") } }); event.target.reset(); announceSession(); notice(""); await workspace(); }); };
    $("logout").onclick = () => action($("logout"), async () => { await api("/logout", { method: "POST" }); announceSession(); state.selected.clear(); state.images = []; $("imageGrid").replaceChildren(); loginView(); });
    $("refresh").onclick = () => action($("refresh"), workspace);
    $("datasetChoice").onchange = () => { state.filters.tranche = ""; const captured = invalidateImages(); history.replaceState(null, "", `/?${new URLSearchParams({dataset: $("datasetChoice").value, view: state.view})}`); return Promise.all([prepareFilters().then(() => currentImages(captured) ? loadImages() : undefined), loadSubmissions()]).catch((error) => { if (!error.stale) notice(error.message, true); }); };
    $("more").onclick = () => action($("more"), () => loadImages(true));
    $("clearSelection").onclick = clearSelection; $("assignee").onchange = selection;
    $("createDataset").onclick = () => $("datasetDialog").showModal();
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    $("datasetForm").onsubmit = (event) => { event.preventDefault(); const form = new FormData(event.currentTarget); action(event.submitter, async () => { const created = await api("/datasets", { method: "POST", body: { name: form.get("name"), classes: String(form.get("classes")).split(/\r?\n/).map((name) => name.trim()).filter(Boolean), operation_id: op() } }); $("datasetDialog").close(); event.target.reset(); state.view = "incoming"; await workspace(); $("datasetChoice").value = created.id; await loadImages(); }); };
    $("assign").onclick = () => window.TatorHostedTranches?.create([...state.selected.values()]);
    $("approveSelected").onclick = () => action($("approveSelected"), async () => {
        const selected = Array.from(state.selected.values());
        if (!confirm(`Approve these ${selected.length} submitted images into the confirmed dataset? Only continue after checking their annotations.`)) return;
        const result = await api(`/datasets/${encodeURIComponent($("datasetChoice").value)}/approve-images`, { method: "POST", body: {
            operation_id: op(), editor_id: op(), images: selected.map((image) => ({ image_id: image.id, expected_version: image.current_version, generation: image.generation, operation_id: op() })),
        } });
        const approved = result.images.filter((image) => image.status === "approved").length;
        const rejected = result.images.filter((image) => image.status !== "approved");
        notice(`${approved} images approved.${rejected.length ? ` ${rejected.length} need another check: ${[...new Set(rejected.map((item) => item.detail || item.error))].join("; ")}` : ""}`, Boolean(rejected.length));
        await loadImages();
    });
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
            if (person.invite_token) passwordLink(person.invite_token, personLabel(person), "invite");
            await afterCommit(person.invite_token ? `Invitation created for ${personLabel(person)}. Copy the link from People and send it privately.` : `${personLabel(person)} can now sign in with the password you set.`, refreshUsers);
        });
    };
    $("copyInvitation").onclick = async () => {
        try { await navigator.clipboard.writeText($("invitationLink").value); notice("Link copied."); }
        catch (_) { $("invitationLink").select(); notice("Select the link and copy it with your keyboard."); }
    };
    $("classes").onclick = () => { state.classDataset = null; $("classList").replaceChildren(); $("classRequestList").replaceChildren(); $("classesDialog").showModal(); notice(""); action($("classes"), loadClasses); };
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
    $("inviteForm").onsubmit = (event) => { event.preventDefault(); const form = new FormData(event.currentTarget); action(event.submitter, async () => { if (form.get("password") !== form.get("repeat")) throw new Error("The passwords do not match."); const token = new URLSearchParams(location.hash.slice(1)).get("invite"); await api("/redeem", { method: "POST", body: { token, password: form.get("password") } }); history.replaceState(null, "", "/"); event.target.reset(); $("invitePanel").hidden = true; notice("Password saved. Sign in with your username."); loginView(); }); };
    window.addEventListener("hashchange", () => { if (!invitationView() && !$("invitePanel").hidden) { $("invitePanel").hidden = true; loginView(); } });
    (async () => { if (invitationView()) return; try { state.session = await api("/session"); await workspace(); } catch (error) { if (!error.stale) { loginView(); if (error.status !== 401) notice(error.message, true); } } })();
    $("imageFilters").onsubmit = event => { event.preventDefault(); state.filters = {q: $("imageSearch").value.trim(), incoming: $("incomingFilter").value, tranche: $("reviewTranche").value, annotator: $("reviewAnnotator").value}; action(event.submitter, () => loadImages()); };
    $("selectVisibleImages").onclick = event => action(event.currentTarget, () => { for (const image of state.images.filter(selectable)) state.selected.set(image.id, image); syncChecks(); });
    $("selectMatchingImages").onclick = event => action(event.currentTarget, selectMatching);
    window.TatorHostedDashboard = { api, action, notice, refresh: workspace,
        selectDataset: async id => { if (!state.datasets.some(d => d.id === id)) throw new Error("Dataset unavailable for this account."); $("datasetChoice").value = id; state.filters.tranche = ""; const captured = invalidateImages(); await prepareFilters(); if (currentImages(captured)) await loadImages(); },
        showView: view, loadImages, clearSelection, get view() { return state.view; }, get users() { return state.users; }, personLabel,
        get session() { return state.session; }, get selectedImages() { return [...state.selected.values()]; }, get datasetId() { return $("datasetChoice").value; } };
})();
