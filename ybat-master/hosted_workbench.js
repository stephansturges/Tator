/* Hosted admin workbench chrome: the full legacy tools under the hosted origin,
   plus the bridge that copies a hosted dataset in and applies label edits back. */
(() => {
    "use strict";
    const boot = window.TATOR_WORKBENCH;
    if (!boot) return;
    const params = new URLSearchParams(location.search);
    let hostedDataset = params.get("dataset") || boot.dataset || "";
    const el = (tag, props = {}, ...children) => { const node = Object.assign(document.createElement(tag), props); node.append(...children); return node; };
    const bar = el("div", {id: "hostedWorkbenchBar"});
    bar.setAttribute("role", "navigation"); bar.setAttribute("aria-label", "Admin workbench");
    const back = el("a", {textContent: "← Admin workspace"});
    const syncBack = () => { back.href = hostedDataset ? `/?${new URLSearchParams({dataset: hostedDataset})}` : "/"; };
    syncBack();
    const dataButton = el("button", {type: "button", id: "hostedWorkbenchDataButton", textContent: "Hosted data"});
    const note = el("span", {id: "hostedWorkbenchNote", textContent: "Full Tator tools on this server. Edits stay in the workbench until you apply them to a hosted dataset."});
    bar.append(back, el("strong", {textContent: "Admin workbench"}), dataButton, note);
    document.body.prepend(bar);
    document.body.classList.add("hosted-workbench");

    // Keep the active tab in the address so a reload returns to it.
    document.addEventListener("click", (event) => {
        const tab = event.target.closest?.(".tab-bar [data-tab]");
        if (!tab) return;
        const next = new URLSearchParams(location.search); next.set("tab", tab.dataset.tab);
        history.replaceState(null, "", `${location.pathname}?${next}`);
    }, true);
    function openTab() {
        const button = document.querySelector(`.tab-bar [data-tab="${CSS.escape(boot.tab || "class-split")}"]`);
        if (button && !button.classList.contains("active")) button.click();
    }
    if (document.readyState === "complete") setTimeout(openTab, 0);
    else window.addEventListener("load", () => setTimeout(openTab, 0), {once: true});

    // ---- hosted API -------------------------------------------------------
    let session = null;
    async function hosted(path, options = {}) {
        const headers = {"Accept": "application/json"};
        if (options.body !== undefined) {
            session = session || await hosted("/api/hosted/session");
            Object.assign(headers, {"Content-Type": "application/json", "x-tator-csrf": session.csrf, "x-tator-instance": session.instance_id});
        }
        const response = await fetch(path, {method: options.method || "GET", headers, credentials: "same-origin",
            body: options.body === undefined ? undefined : JSON.stringify(options.body)});
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            const detail = typeof data.detail === "string" ? data.detail : (data.error || `HTTP ${response.status}`);
            throw new Error(detail);
        }
        return data;
    }
    async function follow(job, onProgress) {
        while (job.state === "running") {
            onProgress(job);
            await new Promise((resolve) => setTimeout(resolve, 1000));
            job = await hosted(`/api/hosted/workbench/jobs/${encodeURIComponent(job.id)}`);
        }
        if (job.state === "failed") throw new Error(job.error || "The job failed.");
        return job.result;
    }
    const when = (seconds) => new Date(seconds * 1000).toLocaleString();
    const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;

    // ---- dialog -----------------------------------------------------------
    const dialog = el("dialog", {id: "hostedWorkbenchData", className: "hosted-wb-dialog"});
    dialog.setAttribute("aria-labelledby", "hostedWorkbenchDataTitle");
    const picker = el("select", {id: "hostedWorkbenchDataset"});
    const makeCopy = el("button", {type: "button", textContent: "Make workbench copy"});
    const copyStatus = el("p", {className: "hosted-wb-status", role: "status"});
    const copies = el("div", {className: "hosted-wb-copies"});
    const review = el("section", {className: "hosted-wb-review", hidden: true});
    const close = el("button", {type: "button", textContent: "Close"});
    dialog.append(
        el("h2", {id: "hostedWorkbenchDataTitle", textContent: "Hosted datasets in the workbench"}),
        el("p", {className: "hosted-wb-help", textContent: "A workbench copy holds the confirmed images and labels of a hosted dataset at this moment. "
            + "Images are linked, not duplicated. Annotators keep working in the hosted dataset. Edits you make in the workbench "
            + "reach the hosted dataset only when you review and apply them here."}),
        el("div", {className: "hosted-wb-row"}, el("label", {htmlFor: "hostedWorkbenchDataset", textContent: "Hosted dataset"}), picker, makeCopy),
        copyStatus, copies, review,
        el("div", {className: "hosted-wb-actions"}, close));
    document.body.append(dialog);
    close.addEventListener("click", () => dialog.close());
    dataButton.addEventListener("click", () => { dialog.showModal(); loadDatasets().catch(showError(copyStatus)); });
    const showError = (target) => (error) => { target.textContent = error.message || String(error); target.dataset.tone = "error"; };
    const say = (target, text, tone = "") => { target.textContent = text; target.dataset.tone = tone; };

    async function loadDatasets() {
        const {datasets} = await hosted("/api/hosted/datasets");
        picker.replaceChildren(...datasets.map((row) => el("option", {value: row.id, textContent: row.name})));
        if (datasets.some((row) => row.id === hostedDataset)) picker.value = hostedDataset;
        hostedDataset = picker.value; syncBack();
        await loadCopies();
    }
    picker.addEventListener("change", () => { hostedDataset = picker.value; syncBack(); review.hidden = true; loadCopies().catch(showError(copyStatus)); });

    async function loadCopies() {
        copies.replaceChildren();
        if (!picker.value) { say(copyStatus, "There are no hosted datasets yet."); return; }
        const {bridges} = await hosted(`/api/hosted/workbench/bridges?${new URLSearchParams({dataset_id: picker.value})}`);
        if (!bridges.length) { say(copyStatus, "This dataset has no workbench copy yet."); return; }
        say(copyStatus, "");
        copies.append(el("h3", {textContent: "Workbench copies"}));
        const list = el("ul");
        for (const bridge of bridges) {
            const button = el("button", {type: "button", textContent: "Review changes"});
            button.addEventListener("click", () => openReview(bridge).catch(showError(copyStatus)));
            const applied = bridge.applied_images ? ` · ${plural(bridge.applied_images, "image")} applied so far` : "";
            const excluded = bridge.excluded_images ? ` · ${plural(bridge.excluded_images, "test image")} not included` : "";
            list.append(el("li", {}, el("span", {}, el("strong", {textContent: bridge.label}),
                el("small", {textContent: `${plural(bridge.image_count, "confirmed image")}${excluded} · made ${when(bridge.created_at)}${applied}`})), button));
        }
        copies.append(list);
    }

    makeCopy.addEventListener("click", async () => {
        if (!picker.value) return;
        makeCopy.disabled = true; review.hidden = true;
        try {
            const job = await hosted(`/api/hosted/datasets/${encodeURIComponent(picker.value)}/workbench/export`, {method: "POST", body: {}});
            const result = await follow(job, (state) => say(copyStatus, state.total ? `Linking images: ${state.done} of ${state.total}…` : "Freezing the confirmed snapshot…"));
            await loadCopies();
            const reload = el("button", {type: "button", textContent: "Reload workbench"});
            reload.addEventListener("click", () => location.reload());
            const excluded = result.excluded_images ? ` ${plural(result.excluded_images, "test-split image")} are not in the copy, because the workbench handles train and val only.` : "";
            copyStatus.replaceChildren(`Copy ready: “${result.label}”.${excluded} Reload the workbench so every tab lists it, then pick it in the tab you need. `, reload);
            copyStatus.dataset.tone = "ok";
        } catch (error) { showError(copyStatus)(error); } finally { makeCopy.disabled = false; }
    });

    async function openReview(bridge) {
        review.hidden = false;
        review.replaceChildren(el("h3", {textContent: `Changes in “${bridge.label}”`}), el("p", {textContent: "Comparing workbench labels with the hosted dataset…"}));
        const changes = await hosted(`/api/hosted/workbench/bridges/${encodeURIComponent(bridge.id)}/changes`);
        const totals = changes.totals, status = el("p", {className: "hosted-wb-status", role: "status"});
        review.replaceChildren(el("h3", {textContent: `Changes in “${bridge.label}”`}));
        if (!totals.images) {
            review.append(el("p", {textContent: "No label edits are waiting. Edits saved in the workbench appear here."})); return;
        }
        review.append(el("p", {textContent: `${plural(totals.images, "edited image")}: ${totals.relabeled} relabeled, ${totals.moved} moved or resized, `
            + `${totals.added} added and ${totals.deleted} deleted objects.` + (totals.conflicts ? ` ${plural(totals.conflicts, "image")} cannot be applied.` : "")}));
        const rows = changes.images.slice(0, 300).map((image) => {
            const stats = image.stats ? `${image.stats.relabeled} relabeled · ${image.stats.moved} moved · ${image.stats.added} added · ${image.stats.deleted} deleted` : "";
            return el("tr", {className: image.conflict ? "conflict" : ""}, el("td", {textContent: image.filename}),
                el("td", {textContent: image.conflict || (image.empty ? `${stats} · now empty` : stats)}));
        });
        review.append(el("div", {className: "hosted-wb-table"}, el("table", {}, el("thead", {}, el("tr", {}, el("th", {textContent: "Image"}), el("th", {textContent: "Change"}))), el("tbody", {}, ...rows))));
        if (changes.images.length > rows.length) review.append(el("p", {textContent: `and ${changes.images.length - rows.length} more images.`}));
        const confirmEmpty = el("input", {type: "checkbox", id: "hostedWorkbenchConfirmEmpty"});
        if (totals.needs_empty_confirmation) {
            review.append(el("p", {className: "hosted-wb-row"}, confirmEmpty, el("label", {htmlFor: "hostedWorkbenchConfirmEmpty",
                textContent: `Also apply ${plural(totals.needs_empty_confirmation, "image")} that now have no objects. They are saved as background images.`})));
        }
        const apply = el("button", {type: "button", className: "primary"});
        const count = () => totals.images - totals.conflicts - (confirmEmpty.checked ? 0 : totals.needs_empty_confirmation);
        const label = () => { apply.textContent = `Apply ${plural(count(), "image")} to the hosted dataset`; apply.disabled = !count(); };
        confirmEmpty.addEventListener("change", label); label();
        apply.addEventListener("click", async () => {
            if (!window.confirm(`Save workbench labels as new confirmed versions of ${plural(count(), "hosted image")}? Earlier versions stay in each image's history.`)) return;
            apply.disabled = true;
            try {
                // Apply exactly what was reviewed: each image's workbench labels as shown above.
                const expected = Object.fromEntries(changes.images.filter((image) => !image.conflict).map((image) => [image.image_id, image.review_sha256]));
                const job = await hosted(`/api/hosted/workbench/bridges/${encodeURIComponent(bridge.id)}/apply`, {method: "POST",
                    body: {operation_id: crypto.randomUUID(), expected, confirm_empty: confirmEmpty.checked}});
                const result = await follow(job, (state) => say(status, state.total ? `Applying: ${state.done} of ${state.total} images…` : "Checking images…"));
                await loadCopies();
                await openReview({...bridge, applied_images: (bridge.applied_images || 0) + result.applied});
                const after = review.querySelector(".hosted-wb-status") || review.appendChild(el("p", {className: "hosted-wb-status", role: "status"}));
                const failed = result.failed.length ? ` ${plural(result.failed.length, "image")} failed: ${result.failed.slice(0, 5).map((row) => `${row.filename} (${row.detail || row.error})`).join(", ")}.` : "";
                const moved = result.changed_since_review ? ` ${plural(result.changed_since_review, "image")} changed after this review, in the workbench or the hosted dataset, and ${result.changed_since_review === 1 ? "was" : "were"} not applied. Review again.` : "";
                say(after, `Applied ${plural(result.applied, "image")}.${result.skipped_empty ? ` Skipped ${plural(result.skipped_empty, "emptied image")}.` : ""}${moved}${failed}`, failed || moved ? "error" : "ok");
            } catch (error) { showError(status)(error); apply.disabled = false; }
        });
        review.append(el("div", {className: "hosted-wb-actions"}, apply), status);
    }
    window.TatorWorkbench = {get hostedDataset() { return hostedDataset; }, bar, dialog};
})();
