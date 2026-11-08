/* Hosted admin workbench chrome: the full legacy tools under the hosted origin,
   plus the bridge that copies a hosted dataset in and applies label edits back. */
(() => {
    "use strict";
    const boot = window.TATOR_WORKBENCH;
    if (!boot) return;
    const params = new URLSearchParams(location.search);
    // The dataset the admin came from; the bar then follows the copy actually open.
    const arrivalDataset = params.get("dataset") || boot.dataset || "";
    let barDataset = arrivalDataset;
    const el = (tag, props = {}, ...children) => { const node = Object.assign(document.createElement(tag), props); node.append(...children); return node; };
    const bar = el("div", {id: "hostedWorkbenchBar"});
    bar.setAttribute("role", "navigation"); bar.setAttribute("aria-label", "Admin workbench");
    const back = el("a", {textContent: "← Admin workspace"});
    const syncBack = () => { back.href = barDataset ? `/?${new URLSearchParams({dataset: barDataset})}` : "/"; };
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
        const response = await nativeFetch(path, {method: options.method || "GET", headers, credentials: "same-origin",
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
    const showError = (target) => (error) => { target.textContent = error.message || String(error); target.dataset.tone = "error"; };
    const say = (target, text, tone = "") => { target.textContent = text; target.dataset.tone = tone; };
    const action = (text, onClick, props = {}) => { const node = el("button", {type: "button", textContent: text, ...props}); node.addEventListener("click", onClick); return node; };
    const bridgeIdOf = (legacyId) => (/^hosted_wb_[A-Za-z0-9]+$/.test(legacyId || "") ? legacyId.slice("hosted_".length) : "");
    // Edits waiting in a copy: those that can be applied, and those that cannot.
    const pendingOf = (changes) => ({applicable: changes.totals.applicable ?? (changes.totals.images - changes.totals.conflicts),
        blocked: changes.totals.conflicts});
    function pendingText(pending) {
        const parts = [pending.applicable && `${plural(pending.applicable, "edited image")} not applied yet`,
            pending.blocked && `${pending.blocked} cannot be applied`].filter(Boolean);
        return parts.join(", ");
    }

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
    dataButton.addEventListener("click", () => { dialog.showModal(); review.hidden = true; loadDatasets().catch(showError(copyStatus)); });

    // The picker only chooses what the dialog lists; the bar and its link keep
    // following the copy open in the workbench.
    async function loadDatasets(preferred = "") {
        const {datasets} = await hosted("/api/hosted/datasets");
        const wanted = preferred || picker.value || barDataset;
        picker.replaceChildren(...datasets.map((row) => el("option", {value: row.id, textContent: row.name})));
        if (datasets.some((row) => row.id === wanted)) picker.value = wanted;
        await loadCopies();
    }
    picker.addEventListener("change", () => { review.hidden = true; loadCopies().catch(showError(copyStatus)); });

    async function loadCopies() {
        copies.replaceChildren();
        if (!picker.value) { say(copyStatus, "There are no hosted datasets yet."); return; }
        const dataset = picker.value;
        const {bridges} = await hosted(`/api/hosted/workbench/bridges?${new URLSearchParams({dataset_id: dataset})}`);
        if (picker.value !== dataset) return;
        if (!bridges.length) { say(copyStatus, "This dataset has no workbench copy yet."); return; }
        say(copyStatus, "");
        copies.append(el("h3", {textContent: "Workbench copies"}));
        const list = el("ul");
        for (const bridge of bridges) {
            const applied = bridge.applied_images ? ` · ${plural(bridge.applied_images, "image")} applied so far` : "";
            const details = el("small", {textContent: `${plural(bridge.image_count, "confirmed image")} · made ${when(bridge.created_at)}${applied}`});
            const row = el("li", {className: bridge.present ? "" : "deleted"}, el("span", {}, el("strong", {textContent: bridge.label}), details));
            if (bridge.present) {
                const pending = el("small", {className: "hosted-wb-pending", textContent: "Checking for edits…"});
                row.firstChild.append(pending);
                // Edits left in older copies stay visible here until applied.
                hosted(`/api/hosted/workbench/bridges/${encodeURIComponent(bridge.id)}/changes`)
                    .then((changes) => { const text = pendingText(pendingOf(changes)); pending.textContent = text || "No edits waiting"; pending.dataset.tone = text ? "pending" : ""; })
                    .catch((error) => { pending.textContent = error.message || String(error); });
                row.append(action("Review changes", () => openReview(bridge).catch(showError(copyStatus))));
            } else {
                row.firstChild.append(el("small", {className: "hosted-wb-pending", textContent: "Deleted in the workbench, with any edits that were not applied."}));
            }
            list.append(row);
        }
        copies.append(list);
    }

    makeCopy.addEventListener("click", async () => {
        if (!picker.value) return;
        makeCopy.disabled = true; review.hidden = true;
        try {
            const dataset = picker.value;
            const result = await makeCopyOf(dataset, copyStatus);
            await loadCopies();
            const openHere = action("Open it here", () => reopenWith(dataset));
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
        if (totals.held) {
            review.append(el("p", {className: "hosted-wb-held", textContent: `${plural(totals.held, "image")} ${totals.held === 1 ? "is" : "are"} open on the annotation canvas right now. `
                + `${totals.held === 1 ? "It is" : "They are"} applied only if closed before you apply; otherwise apply again later.`}));
        }
        const rows = changes.images.slice(0, 300).map((image) => {
            const stats = image.stats ? `${image.stats.relabeled} relabeled · ${image.stats.moved} moved · ${image.stats.added} added · ${image.stats.deleted} deleted` : "";
            const held = image.held_by && !image.conflict ? ` · open on the annotation canvas by ${image.held_by}` : "";
            return el("tr", {className: image.conflict ? "conflict" : held ? "held" : ""}, el("td", {textContent: image.filename}),
                el("td", {textContent: image.conflict || ((image.empty ? `${stats} · now empty` : stats) + held)}));
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
                refreshPending();
                const after = review.querySelector(".hosted-wb-status") || review.appendChild(el("p", {className: "hosted-wb-status", role: "status"}));
                const failed = result.failed.length ? ` ${plural(result.failed.length, "image")} not applied: ${result.failed.slice(0, 5).map((row) => `${row.filename} (${row.detail || row.error})`).join(", ")}.` : "";
                const moved = result.changed_since_review ? ` ${plural(result.changed_since_review, "image")} changed after this review, in the workbench or the hosted dataset, and ${result.changed_since_review === 1 ? "was" : "were"} not applied. Review again.` : "";
                say(after, `Applied ${plural(result.applied, "image")}.${result.skipped_empty ? ` Skipped ${plural(result.skipped_empty, "emptied image")}.` : ""}${moved}${failed}`, failed || moved ? "error" : "ok");
            } catch (error) { showError(status)(error); apply.disabled = false; }
        });
        review.append(el("div", {className: "hosted-wb-actions"}, apply), status);
    }

    // ---- copies ---------------------------------------------------------------
    async function makeCopyOf(datasetId, target) {
        const job = await hosted(`/api/hosted/datasets/${encodeURIComponent(datasetId)}/workbench/export`, {method: "POST", body: {}});
        return follow(job, (state) => say(target, state.total ? `Linking images: ${state.done} of ${state.total}…` : "Freezing the confirmed snapshot…"));
    }
    function reopenWith(datasetId) {
        // A fresh load lists a new copy in every tab; arrive() then opens it.
        const next = new URLSearchParams(location.search); next.set("dataset", datasetId);
        location.replace(`${location.pathname}?${next}`);
    }
    const copyCache = new Map();  // bridge ID -> /bridges/{id} answer
    async function copyInfoFor(bridgeId, {fresh = false} = {}) {
        if (fresh || !copyCache.has(bridgeId)) copyCache.set(bridgeId, hosted(`/api/hosted/workbench/bridges/${encodeURIComponent(bridgeId)}`));
        try { return await copyCache.get(bridgeId); }
        catch (error) { copyCache.delete(bridgeId); throw error; }
    }
    function freshButton(datasetId, datasetName, {changed = false, waitingEdits = () => null} = {}) {
        const node = action(changed ? "Update copy" : "Make fresh copy", async () => {
            // Read at click time: the count follows saves made since the bar was drawn.
            const pending = waitingEdits();
            const waiting = pending?.applicable ? ` ${plural(pending.applicable, "image")} edited in the current copy ${pending.applicable === 1 ? "is" : "are"} not applied yet. `
                + `Apply ${pending.applicable === 1 ? "it" : "them"} first with Review edits; otherwise ${pending.applicable === 1 ? "it stays" : "they stay"} in the current copy, reviewable under Hosted data.` : "";
            if (!window.confirm(`Make a new workbench copy of ${datasetName} from its confirmed images as they are now?${waiting}`)) return;
            node.disabled = true;
            try { await makeCopyOf(datasetId, copyInfo); reopenWith(datasetId); }
            catch (error) { showError(copyInfo)(error); node.disabled = false; }
        });
        if (changed) node.classList.add("hosted-wb-attention");
        return node;
    }

    // ---- the bar follows the open workspace -----------------------------------
    // State: what was arrived for, what the legacy page has open, and the
    // pending edits of that copy.
    const arrival = {dataset: null, bridge: null, skippedNewer: false, gone: false};
    let shown = {key: null}, current = null, pending = null, notice = "", opening = false, renderTicket = 0;
    const pageLoaded = new Promise((resolve) => { if (document.readyState === "complete") resolve(); else window.addEventListener("load", resolve, {once: true}); });
    const legacy = () => window.TatorLegacyWorkbench;

    function openArrivalButton() {
        if (!arrival.bridge || !legacy()) return null;
        return action(`Open ${arrival.dataset.name}`, () => openArrival("any"));
    }
    async function openArrival(replace) {
        const target = arrival.bridge;
        if (!target || !legacy()) return;
        opening = true; notice = ""; say(copyInfo, `Opening ${arrival.dataset.name}…`); copyActions.replaceChildren();
        try {
            if (!await legacy().openDataset(target.legacy_dataset_id, target.label, {replace})) notice = "";
        } catch (error) {
            notice = error?.message || String(error);
        } finally {
            opening = false; render(true);
        }
    }

    async function render(force = false) {
        if (opening || !legacy()) return;
        const opened = legacy().openedDataset() || "";
        const readOnly = Boolean(legacy().readOnly?.());
        const key = `${opened}|${readOnly}|${notice}`;
        if (!force && key === shown.key) return;
        shown.key = key;
        const ticket = ++renderTicket;
        const bridgeId = bridgeIdOf(opened);
        if (bridgeId) {
            let info;
            try { info = await copyInfoFor(bridgeId, {fresh: force}); }
            catch (error) { info = null; }
            if (ticket !== renderTicket) return;
            if (info) { showCopy(info, readOnly); return; }
        }
        current = null; pending = null;
        const label = legacy().openedLabel?.() || "";
        const arrivalName = arrival.dataset?.name;
        const text = opened === "review" ? "A saved review is open here."
            : opened ? `“${label || opened}” is open here. It is not a copy of a hosted dataset, so its edits stay in the workbench.`
            : arrival.gone ? `The workbench copies of ${arrivalName} were deleted in the workbench, with any edits that were not applied.`
            : arrivalName && !notice ? `${arrivalName} is not open here.` : "";
        say(copyInfo, [notice, text].filter(Boolean).join(" "), notice ? "error" : "");
        const buttons = [];
        const open = opened === arrival.bridge?.legacy_dataset_id ? null : openArrivalButton();
        if (open) buttons.push(open);
        if (arrival.dataset) buttons.push(freshButton(arrival.dataset.id, arrival.dataset.name));
        if (opened && arrival.dataset) buttons.forEach((button) => { if (button !== open) { button.disabled = true; button.title = "Open a hosted copy to use the hosted actions."; } });
        copyActions.replaceChildren(...buttons);
        barDataset = arrivalDataset; syncBack();
    }

    function showCopy(info, readOnly) {
        current = info; barDataset = info.dataset_id; syncBack();
        followPickers(info.legacy_dataset_id);
        const moved = info.hosted_changes || {edited: 0, added: 0, removed: 0};
        const changed = moved.edited + moved.added + moved.removed;
        const since = [moved.edited && `${moved.edited} edited`, moved.added && `${moved.added} newly confirmed`,
            moved.removed && `${moved.removed} no longer confirmed`].filter(Boolean).join(", ");
        const parts = [`${info.dataset_name} · workbench copy of ${plural(info.image_count, "confirmed image")} from ${when(info.created_at)}`];
        if (info.newer_copies) parts.push("a newer copy exists under Hosted data");
        else if (info.id === arrival.bridge?.id && arrival.skippedNewer) parts.push("a newer copy was deleted in the workbench");
        if (changed) parts.push(`since then ${since} in the hosted dataset`);
        const holder = legacy().lockHolder?.() || "another editor", mine = /^you\b/.test(holder);
        if (readOnly) parts.push(mine ? "read-only: it is still open in another tab or a closed window" : `read-only: ${holder} is editing it`);
        say(copyInfo, parts.join(" · "), readOnly ? "readonly" : changed ? "stale" : "");
        const buttons = [];
        if (readOnly && mine) {
            buttons.push(action("Take over editing", async (event) => {
                event.currentTarget.disabled = true;
                await legacy().takeOver?.();
                render(true);
            }));
        }
        buttons.push(freshButton(info.dataset_id, info.dataset_name, {changed: Boolean(changed), waitingEdits: () => (current === info ? pending : null)}));
        const reviewEdits = action(reviewLabel(), () => {
            dialog.showModal();
            loadDatasets(info.dataset_id).then(() => openReview(info)).catch(showError(copyStatus));
        });
        reviewEdits.id = "hostedWorkbenchReviewEdits";
        buttons.push(reviewEdits);
        copyActions.replaceChildren(...buttons);
        refreshPending();
    }
    function reviewLabel() {
        if (!pending) return "Review edits";
        if (pending.applicable) return `Review edits (${pending.applicable})`;
        return pending.blocked ? `Review edits (${pending.blocked} cannot apply)` : "Review edits";
    }

    // Pending edits change whenever the workbench saves labels: refresh after
    // saves (and when the page becomes visible), not only on arrival.
    let pendingTimer = 0, pendingInFlight = null;
    function refreshPending() {
        const info = current;
        if (!info || pendingInFlight) return;
        pendingInFlight = hosted(`/api/hosted/workbench/bridges/${encodeURIComponent(info.id)}/changes`).then((changes) => {
            if (current !== info) return;
            pending = pendingOf(changes);
            const button = document.getElementById("hostedWorkbenchReviewEdits");
            if (button) { button.textContent = reviewLabel(); button.title = pendingText(pending) || "No edits waiting"; }
        }).catch(() => {}).finally(() => { pendingInFlight = null; });
    }
    function schedulePendingRefresh() {
        clearTimeout(pendingTimer);
        pendingTimer = setTimeout(() => { if (document.visibilityState === "visible") refreshPending(); }, 1500);
    }
    // Label saves go through the legacy API (Label Images "Save now" and
    // autosave, Data Quality Explorer label changes); watch them finish.
    function nativeFetch(...args) { return originalFetch(...args); }
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (input, init = {}) => {
        const response = await originalFetch(input, init);
        try {
            const method = String(init.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
            const url = new URL(input instanceof Request ? input.url : String(input), location.href);
            if (response.ok && !["GET", "HEAD", "OPTIONS"].includes(method) && url.pathname.startsWith("/workbench/api/")
                && /\/annotation\/|\/class_analysis\//.test(url.pathname)) schedulePendingRefresh();
        } catch (_error) { /* never disturb the legacy request */ }
        return response;
    };
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") { copyCache.clear(); render(true); } });

    // Deleting a copy in Dataset Management deletes its unapplied edits too.
    async function datasetDeleteWarning(entry) {
        const bridgeId = bridgeIdOf(String(entry?.id || ""));
        if (!bridgeId) return "";
        const info = await copyInfoFor(bridgeId, {fresh: true});
        const changes = await hosted(`/api/hosted/workbench/bridges/${encodeURIComponent(bridgeId)}/changes`);
        const waiting = pendingOf(changes), count = waiting.applicable + waiting.blocked;
        if (!count) return `This is a workbench copy of the hosted dataset ${info.dataset_name}. The hosted dataset is not changed.`;
        return `This workbench copy of ${info.dataset_name} has ${plural(count, "edited image")} that ${count === 1 ? "is" : "are"} not applied to the hosted dataset. `
            + `Deleting the copy deletes ${count === 1 ? "that edit" : "those edits"}. To keep ${count === 1 ? "it" : "them"}, cancel and apply ${count === 1 ? "it" : "them"} with Review edits first.`;
    }

    // ---- arriving from the admin view ---------------------------------------
    // The dataset chosen in the admin view opens here, in the tab that was asked
    // for: its newest workbench copy, made on the first visit.
    async function arrive() {
        if (!arrivalDataset) { await pageLoaded; return; }
        const [{datasets}, {bridges}] = await Promise.all([hosted("/api/hosted/datasets"),
            hosted(`/api/hosted/workbench/bridges?${new URLSearchParams({dataset_id: arrivalDataset})}`)]);
        const dataset = datasets.find((row) => row.id === arrivalDataset);
        if (!dataset) { say(copyInfo, "The dataset you came from is not available.", "error"); return; }
        note.hidden = true;
        arrival.dataset = dataset;
        if (!bridges.length) {
            say(copyInfo, `Copying ${dataset.name} into the workbench…`);
            try { await makeCopyOf(dataset.id, copyInfo); }
            catch (error) { notice = error?.message || String(error); await pageLoaded; render(true); return; }
            say(copyInfo, `Opening ${dataset.name}…`);
            reopenWith(dataset.id); return;
        }
        // A copy deleted in Dataset Management is gone; the newest remaining one opens.
        arrival.bridge = bridges.find((bridge) => bridge.present) || null;
        arrival.skippedNewer = Boolean(arrival.bridge) && bridges.indexOf(arrival.bridge) > 0;
        arrival.gone = !arrival.bridge;
        await pageLoaded;
        if (!legacy()) { say(copyInfo, `${dataset.name} · workbench copy from ${arrival.bridge ? when(arrival.bridge.created_at) : "—"}`); return; }
        // Coming from the admin view switches to this dataset, but a reload keeps
        // what is open, and a displayed review is only replaced by the button.
        const arriving = performance.getEntriesByType?.("navigation")?.[0]?.type !== "reload";
        if (arrival.bridge) await openArrival(arriving ? "dataset" : false);
        else render(true);
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
        if (fallbackDataset === datasetId) return;
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

    arrive().catch(showError(copyInfo)).finally(() => {
        // Opening another copy inside the workbench (Dataset Management, Datasets &
        // sessions), taking over or losing a lock: the bar follows within a second.
        setInterval(() => { if (document.visibilityState === "visible") render(); }, 1000);
    });
    window.TatorWorkbench = {get hostedDataset() { return barDataset; }, bar, dialog, datasetDeleteWarning};
})();
