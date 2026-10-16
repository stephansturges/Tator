(() => {
    "use strict";
    const $ = (id) => document.getElementById(id);
    const state = { session: null, sessionEpoch: 0, datasets: [], images: [], selected: new Map(), users: [], view: "confirmed", sequence: 0, workspaceSequence: 0, userSequence: 0, classSequence: 0, submissionSequence: 0, classDataset: null, submissions: [] };
    const op = () => crypto.randomUUID();
    const node = (tag, text, className) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; };
    function notice(message, error = false) {
        for (const element of [$("notice"), ...document.querySelectorAll("dialog[open] [data-dialog-notice]")]) { element.textContent = message; element.dataset.error = String(error); element.hidden = !message; }
    }
    const errorMessages = { dataset_revision_changed: "The dataset changed. Refresh the preview before applying this change.", split_group_conflict: "These images share a capture group. Preview the complete group before changing its split.", editor_lease_conflict: "An image is open for editing. Close its editing tab before changing this selection.", last_administrator: "Keep at least one active administrator.", username_exists: "That username already exists. Refresh People to manage the account or create a password reset link.", class_version_changed: "The class list changed. Refresh classes and review the current names before trying again.", class_request_already_reviewed: "This request has already been reviewed. Refresh classes to see the result.", class_name_exists: "That class name already exists.", user_changed: "This account changed. Refresh People before applying your change.", installation_changed: "The server changed. Sign in again before continuing." };
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
                error.status = response.status;
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
        $("assign").hidden = $("assigneeLabel").hidden = state.view !== "incoming";
        $("approveSelected").hidden = state.view !== "submitted";
        $("assign").disabled = !$("assignee").value;
        window.dispatchEvent(new Event("tator:hosted-selection"));
    }
    function clearSelection() { state.selected.clear(); document.querySelectorAll(".hosted-card input").forEach((input) => { input.checked = false; }); selection(); }
    async function refreshUsers() {
        const sequence = ++state.userSequence, previous = $("assignee").value;
        const result = await api("/users");
        if (sequence !== state.userSequence) return;
        state.users = result.users;
        $("assignee").replaceChildren();
        for (const user of state.users.filter((row) => row.role === "annotator" && !row.disabled)) { const option = node("option", user.username); option.value = user.id; $("assignee").append(option); }
        if (state.users.some((row) => row.id === previous && row.role === "annotator" && !row.disabled)) $("assignee").value = previous;
        selection();
        drawUsers();
    }
    function drawCard(image) {
        const card = node("article", undefined, "hosted-card");
        const link = node("a"); link.href = `/annotate?image=${encodeURIComponent(image.id)}`;
        const photo = node("img"); photo.src = `/api/hosted/images/${encodeURIComponent(image.id)}/media`; photo.alt = image.filename; photo.loading = "lazy";
        const info = node("div", undefined, "card-info"); info.append(node("strong", image.filename));
        const labels = { confirmed: "Confirmed", assigned: "Assigned", returned: "Changes requested", submitted: "Ready for review", unassigned: "Not assigned" };
        info.append(node("small", `${labels[image.state] || image.state}${image.negative ? " · No target objects" : ""}`));
        link.append(photo, info); card.append(link);
        if (state.session.user.role === "admin" && ["unassigned", "submitted", "confirmed"].includes(image.state)) {
            const check = node("input"); check.type = "checkbox"; check.setAttribute("aria-label", `Select ${image.filename}`);
            check.onchange = () => { if (check.checked) state.selected.set(image.id, image); else state.selected.delete(image.id); selection(); }; card.append(check);
        }
        return card;
    }
    async function loadImages(append = false) {
        const sequence = ++state.sequence;
        const id = $("datasetChoice").value;
        if (!append) { clearSelection(); state.images = []; $("imageGrid").replaceChildren(); }
        $("classes").disabled = !id;
        if (!id) { $("empty").hidden = false; $("empty").textContent = state.session.user.role === "admin" ? "Create a dataset, then add images to Incoming." : "No editable work is currently assigned to you."; $("more").hidden = true; return; }
        const query = new URLSearchParams({ dataset_id: id, limit: "100", offset: String(append ? state.images.length : 0) });
        if (state.view !== "mine") query.set("state", state.view);
        const result = await api(`/images?${query}`);
        if (sequence !== state.sequence) return;
        state.images.push(...result.images);
        for (const image of result.images) $("imageGrid").append(drawCard(image));
        $("empty").hidden = state.images.length !== 0;
        $("empty").textContent = state.view === "submitted" ? "No images are waiting for review." : state.view === "confirmed" ? "No confirmed images yet. Add and approve Incoming images to build this dataset." : "No images in this view.";
        $("more").hidden = result.images.length < 100;
    }
    async function view(value) {
        state.view = value;
        const copy = { confirmed: ["Confirmed images", "Approved images used by analysis, training and dataset exports."], incoming: ["Incoming", "Prepare and assign new images. They join the dataset only after approval."], submitted: ["Review", "Check submitted work, return feedback, or approve it into the dataset."], mine: ["My work", "Open an assigned image to annotate it. Submit finished images individually."] }[value];
        $("workspaceTitle").textContent = copy[0]; $("workspaceExplanation").textContent = copy[1];
        for (const button of $("views").children) button.setAttribute("aria-current", button.dataset.view === value ? "page" : "false");
        $("import").hidden = state.session.user.role !== "admin" || value !== "incoming";
        await loadImages();
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
        for (const [key, label] of isAdmin ? [["confirmed", "Dataset"], ["incoming", "Incoming"], ["submitted", "Review"]] : [["mine", "My work"]]) {
            const button = node("button", label); button.type = "button"; button.dataset.view = key; button.onclick = () => action(button, () => view(key)); $("views").append(button);
        }
        const result = await api("/datasets");
        if (sequence !== state.workspaceSequence) return;
        state.datasets = result.datasets;
        const previous = $("datasetChoice").value || new URLSearchParams(location.search).get("dataset"); $("datasetChoice").replaceChildren();
        for (const dataset of state.datasets) { const option = node("option", dataset.name); option.value = dataset.id; $("datasetChoice").append(option); }
        if (state.datasets.some((row) => row.id === previous)) $("datasetChoice").value = previous;
        if (isAdmin) await refreshUsers();
        if (sequence !== state.workspaceSequence) return;
        await view(isAdmin ? (state.view === "mine" ? "confirmed" : state.view) : "mine");
        if (sequence === state.workspaceSequence) window.dispatchEvent(new Event("tator:hosted-workspace"));
    }
    function passwordLink(token, username, reset = false) {
        $("invitationLink").value = `${location.origin}/#invite=${encodeURIComponent(token)}`;
        $("invitationExplanation").textContent = `Send this private link to ${username} yourself. It can be used once and expires after ${reset ? "one hour" : "24 hours"}.${reset ? " Existing sessions have been signed out." : ""}`;
        $("invitation").hidden = false;
    }
    function drawUsers() {
        $("userList").replaceChildren();
        for (const user of state.users) {
            const form = node("form", undefined, "hosted-user"), heading = node("div", undefined, "hosted-user-heading");
            heading.append(node("strong", user.username), node("small", `${user.disabled ? "Disabled" : "Active"}${user.id === state.session.user.user_id ? " · You" : ""}`));
            const fields = node("div", undefined, "hosted-inline-fields"), roleLabel = node("label", "Role"), role = node("select");
            for (const [value, label] of [["annotator", "Annotator"], ["admin", "Administrator"]]) { const option = node("option", label); option.value = value; role.append(option); }
            role.value = user.role; role.setAttribute("aria-label", "Role"); roleLabel.append(role);
            const accessLabel = node("label", undefined, "hosted-checkbox"), disabled = node("input"); disabled.type = "checkbox"; disabled.checked = Boolean(user.disabled);
            accessLabel.append(disabled, document.createTextNode("Disable access")); fields.append(roleLabel, accessLabel);
            const actions = node("div", undefined, "hosted-actions"), save = node("button", "Save account"), reset = node("button", "Password reset link");
            save.type = "submit"; save.disabled = true; reset.type = "button"; reset.disabled = Boolean(user.disabled);
            const changed = () => { save.disabled = role.value === user.role && disabled.checked === Boolean(user.disabled); };
            role.onchange = changed; disabled.onchange = changed;
            form.onsubmit = (event) => {
                event.preventDefault();
                const nextRole = role.value, nextDisabled = disabled.checked;
                const detail = nextDisabled ? "They will lose access immediately. Saved work and assignments will remain." : "Existing sessions will be signed out. Saved work and assignments will remain.";
                if (!window.confirm(`Change ${user.username}'s account? ${detail}`)) return;
                action(save, async () => {
                    await api(`/users/${encodeURIComponent(user.id)}`, { method: "PATCH", body: { role: nextRole, disabled: nextDisabled, expected_role: user.role, expected_disabled: Boolean(user.disabled) } });
                    if (user.id === state.session.user.user_id) { loginView(); notice("Your account changed. Sign in again to continue."); return; }
                    await afterCommit(`Account updated for ${user.username}.`, refreshUsers);
                });
            };
            reset.onclick = () => {
                if (!window.confirm(`Create a password reset link for ${user.username}? Existing sessions will be signed out and previous invitation or reset links will stop working.`)) return;
                action(reset, async () => {
                    const result = await api(`/users/${encodeURIComponent(user.id)}/reset`, { method: "POST" });
                    passwordLink(result.reset_token, user.username, true);
                    notice(`Password reset link created for ${user.username}.`);
                });
            };
            actions.append(save, reset); form.append(heading, fields, actions); $("userList").append(form);
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
    $("datasetChoice").onchange = () => Promise.all([loadImages(), loadSubmissions()]).catch((error) => { if (!error.stale) notice(error.message, true); });
    $("more").onclick = () => action($("more"), () => loadImages(true));
    $("clearSelection").onclick = clearSelection; $("assignee").onchange = selection;
    $("createDataset").onclick = () => $("datasetDialog").showModal();
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    $("datasetForm").onsubmit = (event) => { event.preventDefault(); const form = new FormData(event.currentTarget); action(event.submitter, async () => { const created = await api("/datasets", { method: "POST", body: { name: form.get("name"), classes: String(form.get("classes")).split(/\r?\n/).map((name) => name.trim()).filter(Boolean), operation_id: op() } }); $("datasetDialog").close(); event.target.reset(); state.view = "incoming"; await workspace(); $("datasetChoice").value = created.id; await loadImages(); }); };
    $("assign").onclick = () => action($("assign"), async () => {
        const selected = Array.from(state.selected.values()), expected = {};
        for (const image of selected) expected[image.id] = { expected_version: image.current_version, generation: image.generation };
        await api(`/datasets/${encodeURIComponent($("datasetChoice").value)}/assignments`, { method: "POST", body: { image_ids: selected.map((image) => image.id), annotator_id: $("assignee").value, operation_id: op(), expected } });
        notice(`${selected.length} images assigned.`); await loadImages();
    });
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
    $("personForm").onsubmit = (event) => { event.preventDefault(); const form = new FormData(event.currentTarget); action(event.submitter, async () => { const person = await api("/users", { method: "POST", body: { username: form.get("username"), role: form.get("role") } }); passwordLink(person.invite_token, person.username); event.target.reset(); await afterCommit(`Invitation created for ${person.username}.`, refreshUsers); }); };
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
    window.TatorHostedDashboard = { api, action, notice, refresh: workspace,
        selectDataset: async id => { if (!state.datasets.some(d => d.id === id)) throw new Error("Dataset unavailable for this account."); $("datasetChoice").value = id; await loadImages(); },
        get session() { return state.session; }, get selectedImages() { return [...state.selected.values()]; }, get datasetId() { return $("datasetChoice").value; } };
})();
