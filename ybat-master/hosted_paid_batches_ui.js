/* Canonical remote Batch jobs: immutable inputs, retained proposals, explicit saves. */
(() => {
    "use strict";
    const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
    const operation = () => crypto.randomUUID();
    const dashboard = window.TatorHostedDashboard;
    const base = "/tools/paid-batches";
    const api = (path, body) => dashboard ? dashboard.api(path, body === undefined ? {} : {method: "POST", body})
        : window.TatorHosted.api("/api/hosted" + path, body);
    const response = value => new Response(JSON.stringify(value), {status: 200, headers: {"content-type": "application/json"}});
    const pick = (value, keys) => Object.fromEntries(keys.filter(key => value[key] !== undefined).map(key => [key, value[key]]));
    let dialog, notice, form, list, details, epoch = 0, selected = null, timer = null, datasetId = null;
    const activeDataset = () => dashboard?.datasetId || window.TatorHosted?.datasetId;
    const current = ticket => ticket === epoch && dialog?.open && activeDataset() === datasetId;
    function action(parent, label, callback) {
        const button = node("button", label), ticket = epoch; button.type = "button";
        button.onclick = async () => { if (!current(ticket)) return; button.disabled = true;
            try { await callback(); } catch (error) { if (current(ticket) && !error.stale) notice.textContent = error.message; }
            finally { button.disabled = button.dataset.disabled === "true"; } };
        parent.append(button); return button;
    }
    function ensureDialog() {
        if (dialog) return;
        dialog = node("dialog"); dialog.id = "hostedPaidBatchDialog";
        notice = node("p"); notice.setAttribute("role", "status");
        form = node("section"); list = node("section"); details = node("section");
        const close = node("button", "Close"); close.onclick = () => dialog.close();
        dialog.append(node("h2", "Remote caption batches"), node("p", "Send a frozen set of pictures to the configured provider. This Batch recipe asks for one structured caption and its question-and-answer pairs per picture. It runs remotely; captions stay as proposals until you accept them."), notice, form, list, details, close);
        dialog.addEventListener("close", () => { ++epoch; clearTimeout(timer); }); document.body.append(dialog);
    }
    async function refresh(ticket) {
        const value = await api(`${base}?dataset_id=${encodeURIComponent(datasetId)}`);
        if (!current(ticket)) return;
        list.replaceChildren(node("h3", "Saved jobs"));
        if (!value.jobs.length) list.append(node("p", "No remote batches for this dataset."));
        for (const job of value.jobs) {
            const row = node("div"); row.className = "hosted-analysis-job";
            row.append(node("span", `${job.status.replaceAll("_", " ")} · ${new Date(job.created_at * 1000).toLocaleString()}`));
            action(row, "Inspect", () => inspect(job.job_id, ticket)); list.append(row);
        }
    }
    async function inspect(id, ticket = epoch) {
        clearTimeout(timer); selected = id;
        const job = await api(`${base}/${encodeURIComponent(id)}`);
        if (!current(ticket) || selected !== id) return;
        details.replaceChildren(node("h3", `Batch · ${job.status.replaceAll("_", " ")}`));
        details.append(node("p", `${job.result.accepted_cases || 0} complete caption rows; ${job.result.incomplete_cases || 0} incomplete. No annotation changes are automatic.${job.error ? " " + job.error : ""}`));
        action(details, "Refresh provider status", async () => { await api(`${base}/${id}/poll`, {}); await inspect(id, ticket); });
        if (!["collected", "cancelled", "failed", "expired", "submission_unknown"].includes(job.status)) {
            action(details, "Cancel remote work", async () => { await api(`${base}/${id}/cancel`, {operation_id: operation()}); await inspect(id, ticket); });
            timer = setTimeout(() => inspect(id, ticket).catch(error => { if (current(ticket)) notice.textContent = error.message; }), 3000);
        }
        if (["failed", "interrupted"].includes(job.status)) action(details, "Resume retained work", async () => {
            const context = job.request.image_context && window.TatorHosted ? await window.TatorHosted.capturePaidBatchContext() : {};
            await api(`${base}/${id}/resume`, {operation_id: operation(), ...(context.image_context ? {image_context: context.image_context} : {})}); await inspect(id, ticket);
        });
        if (job.status === "submission_unknown") {
            details.append(node("p", "The server did not receive a reliable submission reply. It will not submit again. Find this job in the provider dashboard and enter its Batch ID; the server checks the recorded input file and job metadata before recovering it."));
            const remote = node("input"); remote.placeholder = "batch_…"; remote.setAttribute("aria-label", "Provider Batch ID"); details.append(remote);
            action(details, "Recover matching batch", async () => { await api(`${base}/${id}/recover`, {operation_id: operation(), remote_id: remote.value.trim()}); await inspect(id, ticket); });
        }
        if (job.result.incomplete_cases) action(details, "Request missing QA only", async () => {
            const original = {...job.request, operation_id: operation(), catchup_job_id: id};
            if (original.image_context) original.image_context = (await window.TatorHosted.capturePaidBatchContext()).image_context;
            const next = await api(base, original); await refresh(ticket); await inspect(next.job_id, ticket);
        });
        action(details, "Archive from this list", async () => { await api(`${base}/${id}/archive`, {operation_id: operation()}); details.replaceChildren(); await refresh(ticket); });
        const gallery = node("div"); gallery.className = "hosted-caption-proposals"; details.append(gallery);
        let offset = 0;
        async function page() {
            const result = await api(`${base}/${id}/captions?offset=${offset}&limit=12`);
            if (!current(ticket) || selected !== id) return;
            gallery.querySelector("button[data-more]")?.remove();
            for (const proposal of result.captions) {
                const card = node("article"), image = node("img");
                image.src = `/api/hosted/images/${encodeURIComponent(proposal.image_id)}/media`; image.loading = "lazy"; image.alt = "Source image";
                card.append(image, node("p", proposal.caption || "No caption collected yet."));
                if (proposal.caption) {
                    card.append(node("small", `${proposal.generated_qa_pairs.length} question-and-answer pairs · ${proposal.quality.final_status}`));
                    if (proposal.generated_qa_pairs.length) {
                        const pairs = node("details"); pairs.append(node("summary", "Read the proposed questions and answers"));
                        for (const pair of proposal.generated_qa_pairs) pairs.append(node("strong", pair.question || "Question"), node("p", pair.answer || ""));
                        card.append(pairs);
                    }
                    if (window.TatorHosted?.imageId === proposal.image_id) action(card, "Use this caption", async () => {
                        await window.TatorHosted.applyPaidCaptionProposal(id, proposal); notice.textContent = "Caption saved. Existing annotations and earlier captions were preserved.";
                    });
                    else { const link = node("a", "Review on image"); link.href = `/annotate?image=${encodeURIComponent(proposal.image_id)}&paid_batch=${encodeURIComponent(id)}`;
                        if (window.TatorHosted) link.onclick = event => { event.preventDefault(); window.TatorHosted.leaveFor(link.href).catch(error => { notice.textContent = error.message; }); };
                        card.append(link); }
                }
                gallery.append(card);
            }
            offset += result.captions.length;
            if (offset < result.total) action(gallery, "Load more", page).dataset.more = "true";
        }
        await page();
    }
    async function open(id = null) {
        ensureDialog(); clearTimeout(timer); const ticket = ++epoch;
        datasetId = activeDataset(); selected = null; notice.textContent = "";
        form.replaceChildren(); list.replaceChildren(); details.replaceChildren(); dialog.showModal();
        if (!datasetId) { notice.textContent = "Choose a dataset first."; return; }
        const catalog = await api(`/tools/catalog?dataset_id=${encodeURIComponent(datasetId)}`);
        if (!current(ticket)) return;
        const admin = (dashboard?.session || window.TatorHosted?.session)?.user.role === "admin";
        if (admin && catalog.paid_batch?.models.length) {
            const label = (title, input) => { const e = node("label", title); e.append(input); form.append(e); return input; };
            const model = label("Batch model", node("select"));
            for (const item of catalog.paid_batch.models) model.add(new Option(item.id, item.id));
            const detail = label("Image detail", node("select")), tokens = label("Maximum output tokens per image", node("input")); tokens.type = "number"; tokens.min = "16";
            const qa = label("QA pairs per image", node("input")); qa.type = "number"; qa.min = "0"; qa.max = "20"; qa.value = "8";
            const effort = label("Reasoning effort", node("select")); for (const value of ["none", "minimal", "low", "medium", "high", "xhigh"]) effort.add(new Option(value, value)); effort.value = "high";
            const questions = label("Required questions, one per line (optional)", node("textarea")); questions.rows = 3;
            const all = label(dashboard ? "Use all confirmed images (otherwise use selected confirmed images)" : "All confirmed images in this dataset", node("input")); all.type = "checkbox"; all.checked = true; all.disabled = !dashboard;
            function changed() { const item = catalog.paid_batch.models.find(row => row.id === model.value); detail.replaceChildren(...item.image_details.map(value => new Option(value, value))); detail.value = item.image_details.includes("high") ? "high" : item.image_details[0]; tokens.max = String(item.max_output_tokens); tokens.value = Math.min(8192, item.max_output_tokens); }
            model.onchange = changed; changed();
            let prepared = null, formRevision = 0;
            const preview = node("details"), previewText = node("pre"); preview.append(node("summary", "Exact first-image prompt"), previewText); form.append(preview);
            const submit = action(form, "Submit this frozen batch", async () => {
                if (!prepared) throw new Error("Preview the batch first.");
                const result = await api(base, prepared); prepared = null; submit.dataset.disabled = "true"; submit.disabled = true; await refresh(ticket); await inspect(result.job_id, ticket);
            }); submit.disabled = true; submit.dataset.disabled = "true";
            const invalidate = () => { ++formRevision; prepared = null; submit.dataset.disabled = "true"; submit.disabled = true; };
            for (const input of [model, detail, tokens, qa, effort, questions, all]) input.addEventListener("input", invalidate);
            action(form, "Freeze inputs and preview cost limit", async () => {
                if (!tokens.reportValidity() || !qa.reportValidity()) return;
                const revision = formRevision;
                const selectedIds = all.checked ? null : dashboard.selectedImages.map(image => image.id);
                const options = {model: model.value, detail: detail.value, max_output_tokens: Number(tokens.value), reasoning_effort: effort.value};
                const settings = {instruction_dataset: Number(qa.value) > 0, subcaptions_per_image: Number(qa.value), instruction_qa_imposed_questions: questions.value.split(/\n/).map(value => value.trim()).filter(Boolean)};
                if (window.TatorHosted) await window.TatorHosted.prepareDatasetCaptionRun();
                if (!current(ticket) || revision !== formRevision) return;
                const snapshot = await api(`/datasets/${datasetId}/snapshots`, {operation_id: operation(), name: "Remote caption batch"});
                if (!current(ticket) || revision !== formRevision) return;
                const request = {dataset_id: datasetId, snapshot_id: snapshot.snapshot_id || snapshot.id, image_ids: selectedIds,
                    operation_id: operation(), provider_options: options, settings, caption_request: {}};
                const result = await api(`${base}/preview`, request);
                if (!current(ticket) || revision !== formRevision) return;
                prepared = request; previewText.textContent = result.full_text; preview.open = true;
                notice.textContent = `${result.image_count} frozen images. The server will reserve at most $${(result.maximum_reserved_microdollars / 1000000).toFixed(4)} using the configured token ceilings. Actual reported usage is settled per image.`;
                submit.dataset.disabled = "false"; submit.disabled = false;
            });
        } else if (admin) form.append(node("p", "Configure a model and explicit Batch prices in Paid services to submit a remote batch."));
        await refresh(ticket); if (id && current(ticket)) await inspect(id, ticket);
    }
    function legacyShape(job) {
        if (job.operation_committed && !job.request) return {...job, message: "The earlier command was committed. Access to its source images has since changed."};
        return {...job, dataset_id: job.request.dataset_id, case_count: job.request.image_context ? 1 : job.result.total_cases || job.request.image_ids?.length || job.request_counts.total || 0,
            status: ({queued: "preparing", validating: "in_progress", finalizing: "in_progress", paused: "preparing", submission_unknown: "unknown", completed: "completed_remote"})[job.status] || job.status,
            openai: job.request.provider_options, message: job.error || "Results remain proposals until you use them on an image."};
    }
    async function legacyFetch(target, options = {}) {
        const suffix = target.pathname.slice("/qwen/caption/openai_batches".length);
        const method = (options.method || "GET").toUpperCase();
        const source = typeof options.body === "string" ? JSON.parse(options.body) : {};
        if (!suffix && method === "GET") {
            if (!activeDataset()) return response({jobs: []});
            const result = await api(`${base}?dataset_id=${encodeURIComponent(activeDataset())}&include_archived=${target.searchParams.get("include_archived") === "true"}`);
            const jobs = await Promise.all(result.jobs.map(async job => ({...legacyShape(job), target_verification: await api(`${base}/${job.job_id}/verify`, {})})));
            return response({jobs});
        }
        if (!suffix && method === "POST") {
            const captured = await window.TatorHosted.capturePaidBatchContext();
            if (source.image_names?.length !== 1 || source.image_names[0].split("/").at(-1) !== captured.image_context.image_id) throw new Error("Submit the open assigned image here. Administrators can use Remote dataset captions in this annotation pane for a dataset run.");
            const catalog = await api(`/tools/catalog?dataset_id=${encodeURIComponent(captured.dataset_id)}`);
            const model = catalog.paid_batch?.models.find(item => item.id === source.openai_model);
            if (!model) throw new Error("This model has no permitted Batch price and token policy for the dataset.");
            const request = {...captured, operation_id: operation(), provider_options: {model: model.id, detail: source.openai_image_detail || "high",
                max_output_tokens: source.caption_request?.max_new_tokens || Math.min(8192, model.max_output_tokens), reasoning_effort: source.openai_reasoning_effort || "high"},
                caption_request: pick(source.caption_request || {}, catalog.caption_job_parameters.caption_request), settings: pick(source, catalog.caption_job_parameters.settings)};
            const result = await api(base, request); return response(legacyShape(result));
        }
        if (suffix === "/scan" || suffix === "/adopt") { await open(); return response({artifacts: [], message: "Recover server-recorded jobs in Remote caption batches using their provider Batch ID. Hosted jobs do not adopt arbitrary server folders."}); }
        const [id, actionName] = suffix.slice(1).split("/");
        if (!id) throw new Error("Batch job is missing.");
        if (!actionName && method === "GET") return response(legacyShape(await api(`${base}/${encodeURIComponent(id)}`)));
        if (actionName === "import") { await open(id); return response({...legacyShape(await api(`${base}/${id}`)), message: "Review the proposed captions and choose Use this caption. Existing annotations are preserved."}); }
        if (actionName === "verify") {
            const report = await api(`${base}/${id}/verify`, {}); return response({ok: report.ok, report});
        }
        if (["poll", "collect", "cancel", "archive", "resume"].includes(actionName)) {
            const command = {operation_id: operation()};
            if (actionName === "archive") command.archived = source.archived !== false;
            if (actionName === "resume") {
                const job = await api(`${base}/${encodeURIComponent(id)}`);
                if (job.request.image_context) command.image_context = (await window.TatorHosted.capturePaidBatchContext()).image_context;
            }
            return response(legacyShape(await api(`${base}/${encodeURIComponent(id)}/${actionName}`, command)));
        }
        throw new Error("This remote Batch action is not available for a hosted job.");
    }
    window.TatorHostedPaidBatches = {legacyFetch, open};
    if (!dashboard && window.TatorHostedBoot?.user.role === "admin") {
        const button = node("button", "Remote dataset captions…"); button.id = "hostedPaidBatchJobs"; button.type = "button"; button.className = "training-button secondary";
        document.getElementById("hostedCaptionJobs")?.insertAdjacentElement("afterend", button);
        button.onclick = () => open().catch(error => { notice.textContent = error.message; });
        window.addEventListener("storage", event => { if (event.key === "tator.hosted.session-event") { button.hidden = true; dialog?.close(); } });
    }
    if (!dashboard) window.addEventListener("load", async () => {
        const id = new URLSearchParams(location.search).get("paid_batch"); if (!id) return;
        const began = Date.now();
        while (!window.TatorHostedCanvas?.ready()) { if (Date.now() - began > 30000) return; await new Promise(resolve => setTimeout(resolve, 100)); }
        open(id).catch(error => { notice.textContent = error.message; });
    }, {once: true});
    window.addEventListener("tator:hosted-image", () => dialog?.close());
})();
