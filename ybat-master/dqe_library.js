(function (root) {
    "use strict";
    const terminal = new Set(["completed", "cancelled", "failed", "partial_failed"]);
    const actions = {
        clear_cache: "Clear reusable cache", clear_analysis_files: "Clear analysis files",
        delete_review: "Permanently delete review", trash_dataset: "Move dataset to trash",
        empty_trash: "Permanently delete selected trash", unlink_dataset: "Unlink dataset",
        remove_abandoned_upload: "Remove abandoned upload", clear_export: "Clear prepared export",
    };
    function create(adapters) {
        const elements = {};
        const state = { context: null, summary: null, items: [], cursor: "", selected: new Set(), generation: 0, cleanupGeneration: 0, preview: null, operation: null, view: "dataset", query: "", parentId: "", busy: false, committing: false, loadingItems: false, legacy: false };
        let searchTimer = null;
        let pollTimer = null;
        let inventoryTimer = null;
        let inventoryFollowup = null;
        const element = (tag, text, className) => {
            const node = root.document.createElement(tag);
            if (text != null) node.textContent = String(text);
            if (className) node.className = className;
            return node;
        };
        const bytes = (value) => value == null || !Number.isFinite(Number(value)) ? "Not measured" : adapters.formatBytes(Number(value));
        function message(text, tone = "") {
            elements.status.textContent = String(text || "");
            elements.status.dataset.tone = tone;
        }
        async function request(path, options = {}, context = state.context) {
            if (!context?.apiRoot) throw new Error("The backend is unavailable.");
            const response = await root.fetch(`${context.apiRoot}/storage/library${path}`, {
                ...options, headers: { ...(options.headers || {}), ...(context.backendId ? { "X-Tator-Backend-Id": context.backendId } : {}) },
            });
            const body = await response.text();
            let payload;
            try { payload = JSON.parse(body || "{}"); } catch (_) { payload = {}; }
            if (!response.ok) {
                const detail = payload.detail;
                const error = new Error(typeof detail === "string" ? detail : detail?.message || detail?.code || `HTTP ${response.status}`);
                error.httpStatus = response.status;
                error.detail = detail;
                throw error;
            }
            if (context.backendId && payload.backend_id && payload.backend_id !== context.backendId) {
                if (state.context?.apiRoot === context.apiRoot && state.context?.backendId === context.backendId) adapters.identityLost?.();
                throw new Error("The server identity changed. Reconnect before editing or managing storage.");
            }
            return payload;
        }
        const json = (method, payload) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        function button(text, callback, disabled = false, title = "") {
            const node = element("button", text, "training-button secondary");
            node.type = "button";
            node.disabled = disabled;
            node.title = title;
            node.addEventListener("click", () => Promise.resolve().then(callback).catch((error) => message(error.message || error, "error")));
            return node;
        }
        function setBusy(busy) {
            state.busy = busy;
            elements.refresh.disabled = busy || state.committing;
            elements.previewButton.disabled = busy || state.committing || state.loadingItems || !state.selected.size || !state.context?.backendId;
            elements.view.disabled = state.committing;
            elements.search.disabled = state.committing;
            elements.action.disabled = state.committing;
            elements.more.disabled = state.committing || state.loadingItems;
            elements.list.querySelectorAll("input[type=checkbox]").forEach((checkbox) => { checkbox.disabled = state.committing; });
            const selected = state.items.filter((item) => state.selected.has(item.id));
            Array.from(elements.action.options).forEach((option) => {
                option.disabled = !selected.length || selected.some((item) => !(item.capabilities?.cleanup_actions || []).includes(option.value));
            });
            if (!elements.action.selectedOptions[0] || elements.action.selectedOptions[0].disabled) elements.action.value = Array.from(elements.action.options).find((option) => !option.disabled)?.value || "";
            elements.previewButton.disabled ||= !elements.action.value;
        }
        function invalidatePreview() {
            state.cleanupGeneration += 1;
            state.preview = null;
            elements.confirm.hidden = true;
            elements.preview.replaceChildren();
        }
        function renderSummary(summary) {
            state.summary = summary;
            const usage = summary.usage || {};
            const locations = summary.locations || [];
            const location = locations.find((item) => item.id === "backend") || locations[0] || {};
            const title = summary.backend_name || location.label || state.context.apiRoot;
            elements.backend.textContent = `Backend — ${title} · ${bytes(usage.logical_bytes)} managed`;
            const parts = locations.map((item) => `${item.label || item.path || "Storage"}: ${bytes(item.free_bytes)} free${item.total_bytes == null ? "" : ` of ${bytes(item.total_bytes)}`}`);
            if (summary.discovery?.status && summary.discovery.status !== "completed") parts.push(`Inventory: ${summary.discovery.status}`);
            if (usage.status && usage.status !== "complete") parts.push(`Measurement: ${usage.status}`);
            elements.volumes.textContent = parts.join(" · ");
            elements.policy.textContent = "Automatic cleanup removes unused reusable cache. Saved datasets and reviews are retained until you explicitly remove them.";
            const cacheBytes = usage.cache_bytes ?? usage.rebuildable_bytes;
            if (cacheBytes != null) elements.policy.textContent += ` Reusable cache: ${bytes(cacheBytes)}.`;
            if (elements.legacyStatus) elements.legacyStatus.textContent = `${bytes(usage.logical_bytes)} managed on backend · saved datasets and reviews retained`;
        }
        function displayItem(item, nested = false) {
            const row = element("article", null, "dqe-library__item");
            row.dataset.itemId = item.id;
            const header = element("div", null, "dqe-library__item-header");
            if (state.view === "cleanup" || state.view === "trash") {
                const checkbox = element("input");
                checkbox.type = "checkbox";
                checkbox.checked = state.selected.has(item.id);
                checkbox.setAttribute("aria-label", `Select ${item.label || item.id}`);
                checkbox.addEventListener("change", () => {
                    if (checkbox.checked) state.selected.add(item.id); else state.selected.delete(item.id);
                    invalidatePreview();
                    setBusy(state.busy);
                });
                header.append(checkbox);
            }
            header.append(element("strong", item.label || item.id));
            const badges = [item.state, item.pinned ? "Keep" : "", item.capabilities?.can_edit ? "Can edit" : item.capabilities?.plot_only ? "Plot only" : ""].filter(Boolean);
            header.append(element("span", badges.join(" · "), "training-help"));
            row.append(header);
            const storage = item.storage || {};
            const summary = item.summary || item;
            const source = item.source || {};
            const scope = summary.analysis_scope || summary.scope;
            const recipe = typeof summary.recipe === "object" ? summary.recipe?.label || summary.recipe?.encoder : summary.recipe;
            const info = [
                summary.image_count != null ? `${Number(summary.image_count).toLocaleString()} images` : "",
                summary.object_count != null ? `${Number(summary.object_count).toLocaleString()} objects` : "",
                scope === "selected_classes" ? (summary.selected_classes || summary.class_names || []).join(", ") : scope === "selected_class" ? summary.class_name || summary.class_names?.[0] : scope === "all_classes" ? "All classes" : "",
                summary.recipe_label || recipe || "",
                `${bytes(storage.logical_bytes)}${storage.status === "partial" ? " measured so far" : " stored"}`,
                storage.unknown_assets ? `${storage.unknown_assets} unmeasured items` : "",
                storage.shared_bytes ? `${bytes(storage.shared_bytes)} shared` : "",
                summary.reviewed_count == null ? (item.kind === "session" ? "Review progress not measured" : "") : `${Number(summary.reviewed_count).toLocaleString()} reviewed`,
                source.version || source.snapshot_id || source.annotation_revision || "",
            ].filter(Boolean);
            row.append(element("div", info.join(" · "), "training-help"));
            const dates = [item.created_at ? `Created ${formatDate(item.created_at)}` : "", item.last_opened_at ? `Opened ${formatDate(item.last_opened_at)}` : "", storage.measured_at ? `Measured ${formatDate(storage.measured_at)}` : ""].filter(Boolean);
            if (dates.length) row.append(element("div", dates.join(" · "), "training-help"));
            const reasons = item.capabilities?.reasons || item.blockers || [];
            if (reasons.length) row.append(element("div", reasons.map((reason) => typeof reason === "string" ? reason : reason.message || reason.code).join(" · "), "training-help"));
            const controls = element("div", null, "dqe-library__actions");
            const caps = item.capabilities || {};
            if (caps.open_review) controls.append(button("Open review", () => openItem(item, "review")));
            if (caps.plot_only) controls.append(button("Open plot only", () => openItem(item, "plot_only")));
            if (caps.open_dataset) controls.append(button("Open dataset", () => openItem(item, "dataset")));
            if (caps.restore && adapters.restoreTrash) controls.append(button("Restore dataset", async () => { await adapters.restoreTrash(item); await refresh(); }));
            if (caps.cleanup_actions?.length) controls.append(button("Manage storage", () => {
                if (state.committing) return;
                state.selected = new Set([item.id]);
                if (!state.items.some((entry) => entry.id === item.id)) state.items.push(item);
                elements.cleanup.hidden = false;
                elements.action.value = caps.cleanup_actions[0];
                invalidatePreview();
                setBusy(state.busy);
                elements.cleanup.scrollIntoView({ block: "nearest" });
            }));
            controls.append(button(item.pinned ? "Unkeep" : "Keep", async () => {
                await request(`/items/${encodeURIComponent(item.id)}`, json("PATCH", { expected_revision: item.revision, pinned: !item.pinned }));
                await refreshItems();
            }));
            controls.append(button("Rename", async () => {
                const label = root.prompt("Dataset or review name", item.label || "");
                if (label == null || !label.trim()) return;
                await request(`/items/${encodeURIComponent(item.id)}`, json("PATCH", { expected_revision: item.revision, label: label.trim() }));
                await refreshItems();
            }));
            const details = element("details", null, "dqe-library__details");
            details.append(element("summary", "Storage details"));
            details.addEventListener("toggle", async () => {
                if (!details.open || details.dataset.loaded) return;
                try {
                    const payload = await request(`/items/${encodeURIComponent(item.id)}`);
                    const detail = payload.item || payload;
                    const dependencies = detail.assets || detail.dependencies || detail.references || [];
                    details.append(element("div", `${bytes(detail.storage?.logical_bytes)} stored · ${bytes(detail.storage?.shared_bytes)} shared. ${dependencies.length} storage items.`, "training-help"));
                    for (const dependency of dependencies) {
                        const label = dependency.label || [dependency.category?.replace(/_/g, " "), dependency.relpath].filter(Boolean).join(" · ") || dependency.role || dependency.asset_id || dependency.id || "Shared storage";
                        const shared = dependency.consumer_count > 1 ? ` · used by ${dependency.consumer_count} library entries` : "";
                        details.append(element("div", `${label}${shared}`, "training-help"));
                    }
                    details.dataset.loaded = "1";
                } catch (error) { message(error.message || error, "error"); }
            });
            row.append(controls, details);
            if (!nested && state.view === "dataset") {
                const reviews = element("details", null, "dqe-library__details");
                reviews.append(element("summary", "Saved reviews"));
                reviews.addEventListener("toggle", async () => {
                    if (!reviews.open || reviews.dataset.loaded) return;
                    const generation = state.generation;
                    try {
                        const payload = await request(`/items?kind=session&parent_id=${encodeURIComponent(item.id)}&limit=30`);
                        if (generation !== state.generation || !reviews.isConnected) return;
                        for (const review of payload.items || []) reviews.append(displayItem(review, true));
                        if (!payload.items?.length) reviews.append(element("div", "No saved reviews for this dataset.", "training-help"));
                        if (payload.next_cursor) reviews.append(button("Show all related reviews", () => {
                            if (state.committing) return;
                            state.view = "session"; elements.view.value = "session";
                            state.parentId = item.id; state.query = ""; elements.search.value = "";
                            state.selected.clear();
                            return refreshItems();
                        }));
                        reviews.dataset.loaded = "1";
                    } catch (error) { message(error.message || error, "error"); }
                });
                row.append(reviews);
            }
            return row;
        }
        function formatDate(value) {
            const date = new Date(typeof value === "number" && value < 1e12 ? value * 1000 : value);
            return Number.isNaN(date.getTime()) ? "Unknown" : date.toLocaleString();
        }
        async function openItem(item, mode) {
            setBusy(true);
            try {
                await adapters.open({ item_id: item.id, expected_revision: item.revision, mode });
                message("Workspace opened.", "success");
                await refreshItems();
            } finally { setBusy(false); }
        }
        async function refreshItems(append = false) {
            if (!state.context?.backendId) return;
            if (append && (state.loadingItems || !state.cursor)) return;
            invalidatePreview();
            const generation = ++state.generation;
            const parameters = new URLSearchParams({ kind: state.view, q: state.query, limit: "30" });
            if (state.parentId) parameters.set("parent_id", state.parentId);
            if (append && state.cursor) parameters.set("cursor", state.cursor);
            state.loadingItems = true;
            setBusy(state.busy);
            try {
                const payload = await request(`/items?${parameters}`);
                if (generation !== state.generation) return;
                state.items = append ? state.items.concat(payload.items || []) : payload.items || [];
                state.cursor = String(payload.next_cursor || "");
                state.selected = new Set(Array.from(state.selected).filter((id) => state.items.some((item) => item.id === id)));
                elements.list.replaceChildren(...state.items.map((item) => displayItem(item)));
                if (!state.items.length) elements.list.append(element("p", "No matching items. Refresh inventory to discover existing datasets and reviews.", "training-help"));
                elements.more.hidden = !state.cursor;
                elements.cleanup.hidden = !["cleanup", "trash"].includes(state.view);
            } finally {
                if (generation === state.generation) { state.loadingItems = false; setBusy(state.busy); }
            }
        }
        async function refreshBrowser() {
            const result = await adapters.store.inventory();
            const estimate = result.estimate;
            elements.browser.textContent = `This browser — ${result.origin} · ${estimate ? `${bytes(estimate.usage)} of ${bytes(estimate.quota)} estimated site quota` : result.estimateError}`;
            elements.browserDetails.replaceChildren(element("p", `${result.records.length} known Tator records · about ${bytes(result.metadataBytes)} serialized metadata. This is part of site usage, not an additional total. Source files and downloaded ZIPs are outside browser storage management.`, "training-help"));
            const source = adapters.browserSource?.();
            if (source?.count) elements.browserDetails.append(element("p", `${source.count.toLocaleString()} selected local files · ${bytes(source.bytes)} source-file size. These files are not a persistent browser dataset cache.`, "training-help"));
            const views = result.records.filter((record) => record.removable);
            if (views.length) elements.browserDetails.append(button("Forget saved plot views", async () => {
                if (!root.confirm("Forget saved plot display preferences in this browser? Pending work, datasets and reviews are kept.")) return;
                for (const record of views) adapters.store.forgetView(record);
                await refreshBrowser();
            }));
        }
        function stopInventoryFollowup() {
            root.clearTimeout(inventoryTimer);
            inventoryFollowup = null;
        }
        function watchInventory(summary) {
            const id = summary.refresh_operation_id;
            if (!id || (inventoryFollowup?.id === id && inventoryFollowup.context === state.context)) return;
            stopInventoryFollowup();
            const followup = { id, context: state.context, generation: summary.discovery?.generation || "", metadataRefreshed: false, pendingRefresh: false, settled: false };
            inventoryFollowup = followup;
            inventoryTimer = root.setTimeout(() => reconcileInventory(followup), 500);
        }
        async function reconcileInventory(followup) {
            const current = () => inventoryFollowup === followup && state.context === followup.context;
            if (!current()) return;
            let summary = null;
            try {
                if (!followup.settled) {
                    const operation = await request(`/operations/${encodeURIComponent(followup.id)}`, {}, followup.context);
                    if (!current()) return;
                    if ((terminal.has(operation.state) && operation.state !== "completed")
                        || (operation.state === "completed" && operation.result?.discovery?.metadata_complete === false)) {
                        stopInventoryFollowup();
                        message("Storage inventory needs attention. Refresh to retry its update.", "warn");
                        return;
                    }
                    summary = await request("/summary", {}, followup.context);
                    if (!current()) return;
                    followup.summary = summary;
                    renderSummary(summary);
                    const discovery = summary.discovery || {};
                    // Metadata is published before large source trees finish
                    // measurement. Do not make restored/trash rows wait for it.
                    const metadataReady = summary.inventory_pending === false || (summary.inventory_pending == null
                        && discovery.metadata_complete && (discovery.generation !== followup.generation || discovery.status === "measuring" || operation.state === "completed"));
                    if (metadataReady && !followup.metadataRefreshed) {
                        followup.metadataRefreshed = true;
                        followup.pendingRefresh = true;
                    }
                    if (operation.state === "completed") {
                        followup.settled = true;
                        followup.pendingRefresh = true;
                    }
                }
                // A preview owns its selection until it is confirmed or changed.
                // Defer the redraw without replacing the cleanup status/journal.
                if (followup.pendingRefresh && !state.preview && !state.busy && !state.committing && !state.loadingItems) {
                    followup.pendingRefresh = false;
                    await refreshItems();
                    if (!current()) return;
                }
                if (followup.settled && !followup.pendingRefresh) {
                    stopInventoryFollowup();
                    const latestSummary = followup.summary;
                    if (latestSummary?.refresh_operation_id && latestSummary.refresh_operation_id !== followup.id) watchInventory(latestSummary);
                    return;
                }
                inventoryTimer = root.setTimeout(() => reconcileInventory(followup), 1500);
            } catch (error) {
                if (!current()) return;
                stopInventoryFollowup();
                message(`Storage inventory could not reconnect: ${error.message || error}. Refresh to retry.`, "warn");
            }
        }
        async function refresh() {
            const requestedContext = state.context;
            setBusy(true);
            try {
                const summary = await request("/summary", {}, requestedContext);
                if (state.context !== requestedContext) return;
                if (Number(summary.api_version) !== 1 || !summary.backend_id) throw new Error("This backend does not support the storage library version required by this page.");
                state.context = { ...state.context, backendId: summary.backend_id };
                state.legacy = false;
                adapters.backendReady(state.context, summary);
                renderSummary(summary);
                elements.controls.hidden = false;
                const readyContext = state.context;
                await Promise.all([refreshItems(), refreshBrowser()]);
                if (state.context !== readyContext) return;
                watchInventory(summary);
                const unfinished = adapters.store.readOperation("cleanup");
                if (unfinished?.operation_id) await pollOperation(unfinished.operation_id);
                else if (unfinished?.idempotency_key && unfinished.preview_id) {
                    const operation = await request("/cleanup/commit", json("POST", unfinished), readyContext);
                    adapters.store.record("cleanup", { ...unfinished, ...operation }, readyContext.backendId);
                    if (state.context === readyContext) await pollOperation(operation.operation_id);
                }
            } catch (error) {
                if (state.context.apiRoot !== requestedContext.apiRoot) return;
                if ([404, 405].includes(error.httpStatus) && !state.context?.backendId) {
                    state.legacy = true;
                    state.context = { ...state.context, backendId: "" };
                    elements.controls.hidden = true;
                    elements.backend.textContent = "Backend update required for datasets and sessions.";
                    message("Latest-session restore and legacy cache controls remain available.");
                    await adapters.legacyRefresh();
                    await refreshBrowser();
                } else message(error.message || error, "error");
            } finally { setBusy(false); }
        }
        async function previewCleanup() {
            const context = state.context;
            const selected = Array.from(state.selected);
            const action = elements.action.value;
            if (!selected.length || !action || state.committing || state.loadingItems) return;
            invalidatePreview();
            const generation = state.cleanupGeneration;
            setBusy(true);
            try {
                const payload = await request("/cleanup/preview", json("POST", { item_ids: selected, action }), context);
                if (state.context !== context || generation !== state.cleanupGeneration) return;
                if (payload.action !== action) throw new Error("The cleanup action changed. Request a new preview before continuing.");
                state.preview = { ...payload, cleanupGeneration: generation };
                elements.preview.replaceChildren(element("strong", `Estimated removable data: ${bytes(payload.estimated_bytes)}`));
                if (payload.items?.length) elements.preview.append(element("p", `Selected: ${payload.items.map((item) => item.label || item.id).join(", ")}`, "training-help"));
                for (const effect of payload.effects || []) elements.preview.append(element("p", typeof effect === "string" ? effect : effect.message || effect.description || effect.code, "training-help"));
                if (payload.retained_assets?.length) elements.preview.append(element("p", `${payload.retained_assets.length} shared or protected storage items will remain.`, "training-help"));
                for (const blocker of payload.blockers || []) elements.preview.append(element("p", blocker.message || blocker.code || String(blocker), "dqe-library__error"));
                elements.preview.append(element("p", "Moving files to trash or quarantine does not free disk space. Actual filesystem free space is measured separately.", "training-help"));
                elements.confirm.hidden = !payload.can_commit;
                elements.confirm.textContent = `Confirm: ${actions[payload.action] || "cleanup"}`;
            } finally { if (state.context === context) setBusy(false); }
        }
        async function commitCleanup() {
            const context = state.context;
            const preview = state.preview;
            if (!preview?.can_commit || state.busy || state.committing || preview.cleanupGeneration !== state.cleanupGeneration) return;
            message("");
            state.committing = true;
            setBusy(true);
            elements.confirm.disabled = true;
            const pending = adapters.store.readOperation("cleanup");
            const operationKey = pending?.preview_id === preview.preview_id && pending.digest === preview.digest && pending.idempotency_key
                ? pending.idempotency_key : root.crypto?.randomUUID?.() || `cleanup-${Date.now()}`;
            try {
                adapters.store.record("cleanup", { idempotency_key: operationKey, preview_id: preview.preview_id, digest: preview.digest }, context.backendId);
                const operation = await request("/cleanup/commit", json("POST", { preview_id: preview.preview_id, digest: preview.digest, idempotency_key: operationKey }), context);
                adapters.store.record("cleanup", { ...operation, preview_id: preview.preview_id, digest: preview.digest, idempotency_key: operationKey }, context.backendId);
                if (state.context !== context) return;
                state.selected.clear();
                invalidatePreview();
                await pollOperation(operation.operation_id);
            } catch (error) { if (state.context === context) message(`Cleanup needs attention: ${error.message || error}`, "error"); }
            finally { if (state.context === context) { state.committing = false; elements.confirm.disabled = false; setBusy(false); } }
        }
        async function pollOperation(id) {
            if (pollTimer) root.clearTimeout(pollTimer);
            const context = state.context;
            const payload = await request(`/operations/${encodeURIComponent(id)}`, {}, context);
            if (state.context !== context) return;
            state.operation = payload;
            const progress = payload.progress;
            const progressText = typeof progress === "number" ? ` · ${Math.round(progress <= 1 ? progress * 100 : progress)}%`
                : progress?.total_assets != null ? ` · ${progress.completed_assets || 0}/${progress.total_assets} storage items processed`
                    : progress?.waiting_for_storage ? " · waiting for active work to release storage" : "";
            elements.operation.textContent = `${payload.type || "Storage operation"}: ${payload.state}${progressText}${payload.error ? ` · ${typeof payload.error === "string" ? payload.error : payload.error.message || payload.error.code}` : ""}`;
            if (terminal.has(payload.state)) {
                if (payload.state === "completed" && adapters.store.readOperation("cleanup")?.operation_id === id) adapters.store.record("cleanup", null, context.backendId);
                const result = payload.result || {};
                if (result.bytes_reclaimed != null) elements.operation.textContent += ` · ${bytes(result.bytes_reclaimed)} removed`;
                else if (result.removed_logical_bytes != null) elements.operation.textContent += ` · ${bytes(result.removed_logical_bytes)} of data removed; filesystem free space is shown above`;
                await adapters.changed();
                if (state.context !== context) return;
                const summary = await request("/summary", {}, context);
                if (state.context !== context) return;
                renderSummary(summary);
                await refreshItems();
                if (state.context !== context) return;
                watchInventory(summary);
                if (["failed", "partial_failed"].includes(payload.state) && ["cleanup", "refresh"].includes(payload.type)) {
                    elements.operation.append(button("Retry remaining work", async () => {
                        await request(`/operations/${encodeURIComponent(id)}/retry`, json("POST", {}), context);
                        if (state.context === context) await pollOperation(id);
                    }));
                }
            } else pollTimer = root.setTimeout(() => pollOperation(id).catch((error) => message(`Reconnect to operation ${id}: ${error.message || error}`, "error")), 1500);
        }
        function init(apiRoot) {
            const ids = { root: "dqeLibrary", status: "dqeLibraryStatus", backend: "dqeLibraryBackend", browser: "dqeLibraryBrowser", volumes: "dqeLibraryVolumes", policy: "dqeLibraryPolicy", browserDetails: "dqeLibraryBrowserDetails", controls: "dqeLibraryControls", refresh: "dqeLibraryRefresh", search: "dqeLibrarySearch", view: "dqeLibraryView", list: "dqeLibraryList", more: "dqeLibraryMore", cleanup: "dqeLibraryCleanup", action: "dqeLibraryCleanupAction", previewButton: "dqeLibraryPreview", preview: "dqeLibraryPreviewDetails", confirm: "dqeLibraryConfirm", operation: "dqeLibraryOperation", legacyStatus: "classAnalysisCacheStatus" };
            Object.entries(ids).forEach(([name, id]) => { elements[name] = root.document.getElementById(id); });
            state.context = { apiRoot, backendId: "" };
            Object.entries(actions).forEach(([value, label]) => { const option = element("option", label); option.value = value; elements.action.append(option); });
            elements.refresh.addEventListener("click", () => refresh());
            root.document.getElementById("dqeLibraryRescan")?.addEventListener("click", async () => {
                const context = state.context;
                try { const operation = await request("/refresh", json("POST", {}), context); if (state.context === context) await pollOperation(operation.operation_id); }
                catch (error) { if (state.context === context) message(error.message || error, "error"); }
            });
            elements.view.addEventListener("change", () => { state.view = elements.view.value; state.parentId = ""; state.selected.clear(); refreshItems().catch((error) => message(error.message || error, "error")); });
            elements.search.addEventListener("input", () => {
                root.clearTimeout(searchTimer);
                state.generation += 1;
                invalidatePreview(); state.selected.clear(); state.cursor = ""; elements.more.hidden = true;
                searchTimer = root.setTimeout(() => { state.parentId = ""; state.query = elements.search.value.trim(); refreshItems().catch((error) => message(error.message || error, "error")); }, 250);
            });
            elements.action.addEventListener("change", invalidatePreview);
            elements.more.addEventListener("click", () => refreshItems(true).catch((error) => message(error.message || error, "error")));
            elements.previewButton.addEventListener("click", () => previewCleanup().catch((error) => message(error.message || error, "error")));
            elements.confirm.addEventListener("click", commitCleanup);
            return refresh();
        }
        return { init, refresh, request, message, getContext: () => state.context, supported: () => Boolean(state.context?.backendId), legacy: () => state.legacy,
            adoptBackend(context, summary) {
                state.generation += 1;
                if (pollTimer) root.clearTimeout(pollTimer);
                stopInventoryFollowup();
                root.clearTimeout(searchTimer);
                state.context = context; state.legacy = false; state.items = []; state.selected.clear(); state.cursor = ""; state.parentId = ""; state.loadingItems = false; state.committing = false;
                invalidatePreview(); elements.list.replaceChildren(); elements.operation.replaceChildren(); elements.more.hidden = true;
                elements.confirm.disabled = false; setBusy(false);
                adapters.backendReady(context, summary); renderSummary(summary);
            },
            show(view = "") {
                elements.root.open = true;
                if (view) { state.view = view; state.parentId = ""; state.selected.clear(); elements.view.value = view; return refreshItems(); }
            },
            async changeBackend(apiRoot) {
                state.generation += 1;
                root.clearTimeout(searchTimer); root.clearTimeout(pollTimer);
                stopInventoryFollowup();
                state.context = { apiRoot, backendId: "" }; state.legacy = false;
                state.items = []; state.selected.clear(); state.cursor = ""; state.parentId = ""; state.loadingItems = false; state.committing = false;
                invalidatePreview(); elements.list.replaceChildren(); elements.operation.replaceChildren(); elements.controls.hidden = true; elements.more.hidden = true;
                elements.confirm.disabled = false;
                adapters.backendReady(state.context);
                return refresh();
            },
        };
    }
    root.TatorDqeLibrary = Object.freeze({ create });
}(typeof globalThis !== "undefined" ? globalThis : this));
