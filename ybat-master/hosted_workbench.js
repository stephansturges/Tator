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
    const copyInfo = el("span", {id: "hostedWorkbenchCopy"}); copyInfo.setAttribute("role", "status");
    const copyActions = el("span", {className: "hosted-wb-copy-actions"});
    bar.append(back, el("strong", {textContent: "Admin workbench"}), copyInfo, copyActions, dataButton, note);
    document.body.prepend(bar);
    document.body.classList.add("hosted-workbench");
    const fitBar = () => document.documentElement.style.setProperty("--hosted-wb-bar", `${bar.offsetHeight}px`);
    fitBar(); if (window.ResizeObserver) new ResizeObserver(fitBar).observe(bar);

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
            list.append(el("li", {}, el("span", {}, el("strong", {textContent: bridge.label}),
                el("small", {textContent: `${plural(bridge.image_count, "confirmed image")} · made ${when(bridge.created_at)}${applied}`})), button));
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
            const dataset = picker.value, openHere = el("button", {type: "button", textContent: "Open it here"});
            openHere.addEventListener("click", () => reopenWith(dataset));
            copyStatus.replaceChildren(`Copy ready: “${result.label}”. `, openHere);
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
    // ---- arriving from the admin view ---------------------------------------
    // The dataset chosen in the admin view opens here, in the tab that was asked
    // for: its newest workbench copy, made on the first visit.
    const pageLoaded = new Promise((resolve) => { if (document.readyState === "complete") resolve(); else window.addEventListener("load", resolve, {once: true}); });
    const action = (text, onClick) => { const node = el("button", {type: "button", textContent: text}); node.addEventListener("click", onClick); return node; };
    async function makeCopyOf(datasetId, target) {
        const job = await hosted(`/api/hosted/datasets/${encodeURIComponent(datasetId)}/workbench/export`, {method: "POST", body: {}});
        return follow(job, (state) => say(target, state.total ? `Linking images: ${state.done} of ${state.total}…` : "Freezing the confirmed snapshot…"));
    }
    function reopenWith(datasetId) {
        // A fresh load lists a new copy in every tab; arrive() then opens it.
        const next = new URLSearchParams(location.search); next.set("dataset", datasetId);
        location.replace(`${location.pathname}?${next}`);
    }
    async function arrive() {
        if (!hostedDataset) return;
        const [{datasets}, {bridges}] = await Promise.all([hosted("/api/hosted/datasets"),
            hosted(`/api/hosted/workbench/bridges?${new URLSearchParams({dataset_id: hostedDataset})}`)]);
        const dataset = datasets.find((row) => row.id === hostedDataset), bridge = bridges[0];
        if (!dataset) return;
        note.hidden = true;
        if (!bridge) {
            say(copyInfo, `Copying ${dataset.name} into the workbench…`);
            await makeCopyOf(dataset.id, copyInfo);
            say(copyInfo, `Opening ${dataset.name}…`);
            reopenWith(dataset.id); return;
        }
        // Hosted edits made after the copy (on the annotation canvas, say) are not in it.
        const moved = bridge.hosted_changes || {edited: 0, added: 0, removed: 0};
        const changed = moved.edited + moved.added + moved.removed;
        const since = [moved.edited && `${moved.edited} edited`, moved.added && `${moved.added} newly confirmed`,
            moved.removed && `${moved.removed} no longer confirmed`].filter(Boolean).join(", ");
        const describe = () => {
            say(copyInfo, `${dataset.name} · workbench copy of ${plural(bridge.image_count, "confirmed image")} from ${when(bridge.created_at)}`
                + (changed ? ` · since then ${since} in the hosted dataset` : ""));
            copyInfo.dataset.tone = changed ? "stale" : "";
        };
        let pending = 0;
        const fresh = action(changed ? "Update copy" : "Make fresh copy", async () => {
            const waiting = pending ? ` ${plural(pending, "image")} edited in the current copy ${pending === 1 ? "is" : "are"} not applied yet. Apply ${pending === 1 ? "it" : "them"} first with Review edits; otherwise ${pending === 1 ? "it stays" : "they stay"} in the current copy, reviewable under Hosted data.` : "";
            if (!window.confirm(`Make a new workbench copy of ${dataset.name} from its confirmed images as they are now?${waiting}`)) return;
            fresh.disabled = true;
            try { await makeCopyOf(dataset.id, copyInfo); reopenWith(dataset.id); }
            catch (error) { showError(copyInfo)(error); fresh.disabled = false; }
        });
        const reviewEdits = action("Review edits", () => {
            dialog.showModal(); hostedDataset = dataset.id;
            loadDatasets().then(() => openReview(bridge)).catch(showError(copyStatus));
        });
        if (changed) fresh.classList.add("hosted-wb-attention");
        copyActions.replaceChildren(fresh, reviewEdits);
        hosted(`/api/hosted/workbench/bridges/${encodeURIComponent(bridge.id)}/changes`).then((result) => {
            pending = result.totals.images;
            if (pending) reviewEdits.textContent = `Review edits (${pending})`;
        }).catch(() => {});
        await pageLoaded;
        const legacy = window.TatorLegacyWorkbench;
        if (!legacy) { describe(); return; }
        // Coming from the admin view switches to this dataset, but a reload keeps
        // what is open, and a displayed review is only replaced by the button.
        const arriving = performance.getEntriesByType?.("navigation")?.[0]?.type !== "reload";
        const openButton = action(`Open ${dataset.name}`, () => open("any"));
        async function open(replace) {
            say(copyInfo, `Opening ${dataset.name}…`); openButton.remove();
            try {
                if (await legacy.openDataset(bridge.legacy_dataset_id, bridge.label, {replace})) { describe(); followPickers(bridge.legacy_dataset_id); return; }
                say(copyInfo, "Another dataset or saved review is open here.");
            } catch (error) { showError(copyInfo)(error); }
            copyActions.prepend(openButton);
        }
        await open(arriving ? "dataset" : false);
    }

    // ---- dataset pickers follow the open dataset ---------------------------
    // Tabs with their own dataset list (training, recipes, captions, ingestion
    // reference...) select the dataset open in the workbench until you pick
    // another one there yourself.
    const PICKERS = ["qwenCaptionDataset", "dataIngestionReferenceDataset", "qwenAgentDataset", "qwenCalibrationDataset", "trainDatasetSelect",
        "qwenDatasetSelect", "sam3DatasetSelect", "yoloDatasetSelect", "yoloHeadGraftDataset", "rfdetrDatasetSelect", "agentDatasetSelect",
        "promptHelperDatasetSelect", "datasetGlossaryDataset", "segBuilderDatasetSelect"];
    const pickedHere = new Set();
    let fallbackDataset = "";
    function syncPickers() {
        const open = window.TatorLegacyWorkbench?.openedDataset?.() || "";
        const target = open && !["transient", "review"].includes(open) ? open : fallbackDataset;
        if (!target) return;
        for (const id of PICKERS) {
            const select = document.getElementById(id);
            if (!select || pickedHere.has(id) || select.value === target || ![...select.options].some((option) => option.value === target)) continue;
            select.value = target;
            select.dispatchEvent(new Event("change", {bubbles: true}));
        }
    }
    function followPickers(datasetId) {
        fallbackDataset = datasetId;
        for (const id of PICKERS) {
            const select = document.getElementById(id);
            if (!select || select.dataset.hostedFollow) continue;
            select.dataset.hostedFollow = "1";
            select.addEventListener("change", (event) => { if (event.isTrusted) pickedHere.add(id); });
            new MutationObserver(syncPickers).observe(select, {childList: true});
        }
        syncPickers();
    }
    document.addEventListener("click", (event) => {
        if (!fallbackDataset || !event.target.closest?.(".tab-bar [data-tab]")) return;
        for (const delay of [400, 2000]) setTimeout(syncPickers, delay);
    }, true);
    arrive().catch(showError(copyInfo));
    window.TatorWorkbench = {get hostedDataset() { return hostedDataset; }, bar, dialog};
})();
