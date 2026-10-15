/* Native DQE projections and evidence; hosted leases protect every correction. */
(() => {
    "use strict";
    const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
    let plotly;
    function loadPlotly() {
        if (window.Plotly) return Promise.resolve(window.Plotly);
        if (!plotly) plotly = new Promise((resolve, reject) => {
            const script = node("script"); script.src = "/plotly-2.35.2.min.js";
            script.onload = () => resolve(window.Plotly); script.onerror = () => { plotly = null; reject(new Error("Could not load the graph. Reopen this result to retry.")); }; document.head.append(script);
        });
        return plotly;
    }
    window.TatorHostedDQE = async ({app, jobId, datasetId, container, valid, report}) => {
        const base = `/analysis/jobs/${encodeURIComponent(jobId)}/dqe`, op = () => crypto.randomUUID();
        const api = (path, body) => app.api(path, body === undefined ? {} : {method: "POST", body});
        const [Plotly, initial, dataset, review, catalog] = await Promise.all([loadPlotly(), api(base + "/graph?limit=5000"), api(`/datasets/${datasetId}`), api(base + "/review"), api(`/tools/catalog?dataset_id=${datasetId}`)]);
        if (!valid()) return;
        const root = node("section"); root.className = "hosted-dqe"; container.append(root);
        const message = node("p"), filters = node("div"), layout = node("div"), plot = node("div"), detail = node("aside"), selection = node("section"), pages = node("div"), batches = node("section");
        filters.className = "hosted-inline-fields"; layout.className = "hosted-dqe-layout"; plot.className = "hosted-dqe-plot"; plot.setAttribute("aria-label", "Object feature graph"); message.setAttribute("role", "status");
        detail.append(node("p", "Click a dot to inspect an object. Drag with the box or lasso tool to select several objects."));
        layout.append(plot, detail); root.append(node("p", "Each dot is a labelled object. Nearby dots have similar features. The graph stays fixed to this analysis; corrections create new annotation versions and preserve the originals."), message, filters, layout, pages, selection, batches);
        let state = review, graph = initial, columns = {}, selected = [], cursor = null, request = 0, pointRequest = 0, busy = false;
        const editor = `dqe-${op()}`, leases = new Map();
        const journalKey = `tator.dqe.action:${app.session.instance_id}:${app.session.user.user_id}:${jobId}`;
        let pending = JSON.parse(sessionStorage.getItem(journalKey) || "null");
        const recovery = node("section"); root.prepend(recovery);
        const classes = dataset.classes.filter(c => !c.retired);
        function label(parent, text, element) { const l = node("label", text); element.setAttribute("aria-label", text); l.append(element); parent.append(l); return element; }
        function choose(parent, text, values) { const s = node("select"); for (const [value, title] of values) s.add(new Option(title, value)); return label(parent, text, s); }
        function action(parent, title, callback, mutation = false) {
            const b = node("button", title); b.type = "button"; b.disabled = mutation && state.source_changed;
            b.onclick = async () => { if (busy || !valid()) return; if (mutation && pending) { message.textContent = "Check the previous correction's result before making another change."; return; } b.disabled = true; try { await callback(); } catch (e) { if (valid() && !e.stale) report(e); } finally { if (valid()) b.disabled = mutation && state.source_changed; } }; parent.append(b); return b;
        }
        const mode = choose(filters, "Feature view", initial.available_projection_modes.map(v => [v, v.replaceAll("_", " ")])); mode.value = initial.projection_mode;
        const category = choose(filters, "Objects", [["all", "All objects"], ["review", "Suggested for review"], ["wrong_class", "Possible wrong class"], ["spatial_evidence", "Feature outliers"], ["overlap", "Overlapping boxes"]]);
        const className = choose(filters, "Class", [["__all__", "All classes"], ...Object.keys(initial.class_counts).sort().map(c => [c, c])]);
        const size = choose(filters, "Size", [["all", "All sizes"], ["tiny_only", "Tiny or low-detail objects"], ["exclude_tiny", "Exclude tiny objects"], ["relative_small", "Small within the picture"]]);
        const reviewed = choose(filters, "Review status", [["any", "All"], ["unreviewed", "Not reviewed"], ["reviewed", "Reviewed"]]);
        function status() { message.textContent = state.source_changed ? "Another edit changed this dataset. You can inspect these saved results; run a new analysis before applying corrections." : `${columns.point_id.length.toLocaleString()} of ${graph.total_matching.toLocaleString()} matching objects shown. ${selected.length} selected.`; }
        async function draw(data, append = false) {
            graph = data; cursor = data.next_cursor;
            columns = append ? Object.fromEntries(Object.keys(data.columns).map(k => [k, [...columns[k], ...data.columns[k]]])) : data.columns;
            selected = []; renderSelection(); status();
            const names = [...new Set(columns.class_name)];
            const traces = names.map((name, color) => {
                const rows = columns.point_id.map((_, i) => i).filter(i => columns.class_name[i] === name);
                return {type: "scattergl", mode: "markers", name, x: rows.map(i => columns.x[i]), y: rows.map(i => columns.y[i]), customdata: rows.map(i => columns.point_id[i]), text: rows.map(i => `${name}${columns.reviewed[i] ? " · reviewed" : ""}`), hovertemplate: "%{text}<extra></extra>", marker: {size: 7, opacity: .75, color: `hsl(${color * 137.5 % 360},60%,48%)`}};
            });
            const css = getComputedStyle(document.documentElement);
            await Plotly.react(plot, traces, {height: 500, margin: {l: 35, r: 10, t: 10, b: 35}, dragmode: "lasso", uirevision: `${mode.value}:${className.value}:${category.value}:${size.value}:${reviewed.value}`, paper_bgcolor: css.getPropertyValue("--surface"), plot_bgcolor: css.getPropertyValue("--surface"), font: {color: css.getPropertyValue("--text")}, xaxis: {title: "Feature direction 1", zeroline: false}, yaxis: {title: "Feature direction 2", zeroline: false}, legend: {orientation: "h"}}, {responsive: true, scrollZoom: true, displaylogo: false, modeBarButtonsToRemove: ["toImage"]});
            if (!valid()) return;
            pages.replaceChildren();
            if (cursor) action(pages, "Load more matching objects", () => refresh(true));
        }
        async function refresh(append = false) {
            const turn = ++request; ++pointRequest;
            const q = new URLSearchParams({limit: "5000", projection_mode: mode.value, class_name: className.value, objects: category.value, object_size: size.value, reviewed: reviewed.value});
            if (append && cursor) q.set("cursor", cursor);
            const [data, nextState] = await Promise.all([api(`${base}/graph?${q}`), api(base + "/review")]);
            if (!valid() || turn !== request) return; state = nextState; await draw(data, append);
        }
        for (const control of filters.querySelectorAll("select")) control.onchange = () => refresh().catch(e => { if (valid()) report(e); });
        async function releaseAll() {
            for (const [id, lease] of leases) {
                if (pending?.context.image_id === id) continue;
                try { await api(`/images/${id}/release`, {expected_version: lease.current_version, generation: lease.generation, editor_id: lease._editor_id || editor, operation_id: op()}); }
                catch (e) { if (valid() && !e.stale) message.textContent = "Changes were retained. An editing reservation could not be released; it will expire automatically."; }
            }
            leases.clear();
        }
        function clearPending() { sessionStorage.removeItem(journalKey); pending = null; recovery.replaceChildren(); }
        function showPending() {
            recovery.replaceChildren(); if (!pending) return;
            recovery.append(node("p", "A correction was sent without a confirmed reply. Check its saved result before making another change."));
            action(recovery, "Check correction result", async () => {
                const command = pending;
                const receipt = await api(`/receipts/${encodeURIComponent(command.context.operation_id)}`);
                let result = receipt?.result;
                if (!result) {
                    try { result = await api(base + "/actions", command); }
                    catch (e) { if ([400, 409, 422].includes(e.status)) { clearPending(); await refresh(); } throw e; }
                }
                clearPending();
                leases.set(result.image.id, {...result.image, _editor_id: command.context.editor_id});
                await releaseAll();
                if (valid()) { await refresh(); detail.replaceChildren(node("p", "The previous correction was saved.")); message.textContent = "Correction confirmed. You can continue reviewing."; }
            });
        }
        async function inspect(id) {
            const turn = ++pointRequest;
            const value = await api(`${base}/points/${encodeURIComponent(id)}`);
            if (!valid() || turn !== pointRequest) return;
            const identity = value.hosted_identity, actualClass = classes.find(c => c.id === identity.class_id);
            const crop = node("img"); crop.src = `/api/hosted${base}/points/${encodeURIComponent(id)}/image`; crop.alt = actualClass?.name || value.point.class_name;
            const full = node("img"); full.src = crop.src + "?context=true&size=512"; full.alt = "Full image containing this object";
            const open = node("a", "Open image and annotations →"); open.href = `/annotate?image=${encodeURIComponent(identity.image_id)}`;
            detail.replaceChildren(node("h3", actualClass?.name || value.point.class_name), crop, full, open);
            if (identity.deleted) { detail.append(node("p", "This object was removed in a saved correction.")); return; }
            const note = label(detail, "Review note (optional)", node("textarea")); note.rows = 2;
            for (const [disposition, title] of [["keep", "Looks correct"], ["uncertain", "Needs another look"]]) action(detail, title, async () => {
                const result = await api(`${base}/points/${encodeURIComponent(id)}/review`, {disposition, note: note.value, expected_revision: state.review_revision, operation_id: op()});
                if (valid()) { state.review_revision = result.review_revision; await refresh(); message.textContent = "Review saved."; }
            }, true);
            const target = choose(detail, "Correct class", classes.map(c => [c.id, c.name])); target.value = identity.class_id;
            action(detail, "Save class correction", () => apply([id], "relabel", target.value), true);
            action(detail, "Remove this annotation", () => apply([id], "delete"), true);
            const reviewers = Object.entries(catalog.heavy_job_options || {}).filter(([, v]) => v.operation === "dqe_review");
            if (reviewers.length) {
                const reviewer = choose(detail, "Visual reviewer", reviewers.map(([key, v]) => [key, v.label || key]));
                action(detail, "Ask Qwen to review this object", async () => {
                    const definition = catalog.heavy_job_options[reviewer.value];
                    const run = await api("/tools/heavy", {dataset_id: datasetId, job: reviewer.value, operation_id: op(), parameters: {...definition.defaults, parent_analysis_job_id: jobId, point_id: id}});
                    if (!valid() || turn !== pointRequest) return;
                    const progress = node("p", "Qwen review queued. Annotation assistance pauses while this analysis runs."), results = node("section"); detail.append(progress, results);
                    const cancel = action(detail, "Cancel Qwen review", async () => { await api(`/tools/requests/${run.id}/cancel`, {}); });
                    let status = run;
                    while (!["completed", "failed", "cancelled", "interrupted", "stale", "unavailable"].includes(status.state)) {
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        if (!valid() || turn !== pointRequest) return;
                        status = await api(`/tools/requests/${run.id}`); progress.textContent = `Qwen review: ${status.state.replaceAll("_", " ")}`;
                    }
                    cancel.remove();
                    if (status.state !== "completed") { progress.textContent += status.error ? ` · ${status.error}` : ""; return; }
                    const saved = await api(`/analysis/jobs/${status.analysis_job_id}/result`);
                    if (!valid() || turn !== pointRequest) return;
                    const native = saved.result?.engine_result || saved.result || saved;
                    progress.textContent = "Review complete. This is a proposal; use the correction buttons only after checking it.";
                    const result = native.result || native.review?.result || {};
                    for (const key of ["decision", "recommendation", "proposed_class", "reason", "reasoning", "rationale"]) if (typeof result[key] === "string") results.append(node("p", `${key.replaceAll("_", " ")}: ${result[key]}`));
                    const evidence = node("details"); evidence.append(node("summary", "Complete model result and reasoning trace"), node("pre", JSON.stringify(native, null, 2))); evidence.className = "hosted-dqe-evidence"; results.append(evidence);
                }, true);
            }
            const evidence = node("details"); evidence.append(node("summary", "Inspect saved comparison evidence")); detail.append(evidence);
            evidence.ontoggle = async () => {
                if (!evidence.open || evidence.dataset.loaded) return; evidence.dataset.loaded = "true";
                try { const response = await api(`${base}/points/${encodeURIComponent(id)}/evidence`); if (!valid() || turn !== pointRequest) return;
                    const pre = node("pre", JSON.stringify(response.evidence, null, 2)); pre.className = "hosted-dqe-evidence"; evidence.append(pre);
                } catch (e) { if (valid() && turn === pointRequest) evidence.append(node("p", e.status === 404 ? "No detailed comparison was saved for this object." : e.message)); }
            };
        }
        function renderSelection() {
            selection.replaceChildren(node("h3", `${selected.length} selected objects`)); if (!selected.length) return;
            const target = choose(selection, "New class for selected objects", classes.map(c => [c.id, c.name]));
            action(selection, "Relabel selected objects", () => apply([...selected], "relabel", target.value), true);
            action(selection, "Remove selected annotations", () => apply([...selected], "delete"), true);
            action(selection, "Clear selection", () => { selected = []; renderSelection(); status(); });
        }
        async function processBatch(batchId) {
            let result;
            const cancel = node("button", "Cancel remaining corrections");
            cancel.onclick = async () => {
                if (!valid()) return; cancel.disabled = true;
                try { await api(`${base}/batches/${batchId}/cancel`, {operation_id: op()}); }
                catch (e) { if (valid() && !e.stale) report(e); cancel.disabled = false; }
            };
            message.after(cancel);
            try {
            do {
                if (!valid()) return;
                result = await api(`${base}/batches/${batchId}/process`, {limit: 5});
                for (const value of Object.values(result.results)) if (value.result?.image) leases.set(value.result.image.id, value.result.image);
                message.textContent = `${Object.keys(result.results).length} of ${result.items.length} image corrections processed.`;
            } while (!["complete", "cancelled", "needs_attention"].includes(result.state));
            const failures = Object.values(result.results).filter(v => v.error);
            if (valid()) await refresh();
            if (failures.length && result.state !== "cancelled") throw new Error(`${failures.length} image correction(s) need attention. Saved corrections are retained; inspect the batch below to continue.`);
            if (valid()) {
                const completed = Object.values(result.results).filter(v => v.state === "complete").length;
                detail.replaceChildren(node("p", result.state === "cancelled" ? "The remaining corrections were cancelled. Earlier saved corrections remain." : "Corrections saved. Select an object to continue reviewing."));
                message.textContent = result.state === "cancelled" ? `${completed} of ${result.items.length} image corrections saved; the rest were cancelled.` : "Corrections saved as new annotation versions. Original images are unchanged.";
            }
            return result;
            } finally { cancel.remove(); }
        }
        async function loadBatches() {
            const value = await api(base + "/batches"); if (!valid()) return;
            batches.replaceChildren(); if (!value.batches.length) return;
            batches.append(node("h3", "Saved bulk corrections"));
            for (const batch of value.batches) {
                const row = node("div"); row.className = "hosted-analysis-job";
                row.append(node("span", `${batch.completed_count} of ${batch.image_count} images saved · ${batch.state.replaceAll("_", " ")}`));
                if (batch.state !== "complete") {
                    action(row, "Resume corrections", async () => {
                        busy = true;
                        try {
                            const retained = await api(`${base}/batches/${batch.batch_id}`), contexts = [];
                            for (const [index, item] of retained.items.entries()) {
                                if (retained.results[index]?.state === "complete") continue;
                                if (!valid()) return;
                                const image = await api(`/images/${item.context.image_id}`);
                                const lease = await api(`/images/${image.id}/editor`, {editor_id: editor, expected_version: image.current_version, generation: image.generation}); leases.set(image.id, lease);
                                contexts.push({image_id: image.id, expected_version: lease.current_version, generation: lease.generation, editor_id: editor, operation_id: op()});
                            }
                            await api(`${base}/batches/${batch.batch_id}/resume`, {contexts, operation_id: op()});
                            await processBatch(batch.batch_id);
                        } finally { await releaseAll(); busy = false; await loadBatches(); }
                    }, true);
                    action(row, "Cancel remaining corrections", async () => { await api(`${base}/batches/${batch.batch_id}/cancel`, {operation_id: op()}); await loadBatches(); }, true);
                }
                batches.append(row);
            }
        }
        async function apply(ids, kind, targetClass) {
            if (!ids.length || state.source_changed) return;
            const title = kind === "delete" ? `Remove ${ids.length} selected annotation(s)? Images and previous annotation versions are preserved.` : `Change ${ids.length} selected annotation(s) to ${classes.find(c => c.id === targetClass)?.name}?`;
            if (!confirm(title)) return;
            busy = true; const items = [], byImage = new Map();
            try {
                for (const id of ids) {
                    if (!valid()) return;
                    const point = await api(`${base}/points/${encodeURIComponent(id)}`), identity = point.hosted_identity;
                    if (identity.deleted) throw new Error("A selected object has already been removed. Refresh the graph.");
                    if (!byImage.has(identity.image_id)) byImage.set(identity.image_id, {identity, actions: []});
                    byImage.get(identity.image_id).actions.push({point_id: id, action: kind, ...(targetClass ? {target_class_id: targetClass} : {})});
                }
                for (const [id, group] of byImage) {
                    if (!valid()) return;
                    const image = await api(`/images/${id}`);
                    if (image.current_version !== group.identity.version_id) throw new Error("An image changed. Refresh before applying this correction.");
                    const lease = await api(`/images/${id}/editor`, {editor_id: editor, expected_version: image.current_version, generation: image.generation}); leases.set(id, lease);
                    const negative = kind === "delete" && group.actions.length === image.annotations.objects.length;
                    if (negative && !confirm(`This removes every annotation from ${image.filename}. Confirm that it contains no target objects?`)) return;
                    items.push({context: {image_id: id, expected_version: lease.current_version, generation: lease.generation, editor_id: editor, operation_id: op()}, actions: group.actions, negative});
                }
                if (!valid()) return;
                if (items.length === 1) {
                    const command = {...items[0], expected_review_revision: state.review_revision};
                    sessionStorage.setItem(journalKey, JSON.stringify(command)); pending = command; showPending();
                    try {
                        const result = await api(base + "/actions", command);
                        clearPending(); leases.set(result.image.id, result.image);
                    } catch (e) { if ([400, 409, 422].includes(e.status)) clearPending(); throw e; }
                } else {
                    const batch = await api(base + "/batches", {items, operation_id: op()});
                    const outcome = await processBatch(batch.batch_id);
                    if (!outcome || outcome.state !== "complete") return;
                }
                if (valid()) { await refresh(); detail.replaceChildren(node("p", "Correction saved. Select an object to continue reviewing.")); message.textContent = "Corrections saved as new annotation versions. Original images are unchanged."; }
            } finally { await releaseAll(); busy = false; if (valid()) await loadBatches(); }
        }
        await draw(initial);
        showPending();
        plot.on("plotly_click", event => { const id = event.points?.[0]?.customdata; if (id) inspect(id).catch(e => { if (valid()) report(e); }); });
        plot.on("plotly_selected", event => { selected = [...new Set((event?.points || []).map(p => p.customdata))]; renderSelection(); status(); });
        await loadBatches();
        return () => { ++request; ++pointRequest; Plotly.purge(plot); };
    };
    window.TatorHostedDQE.loadPlotly = loadPlotly;
})();
