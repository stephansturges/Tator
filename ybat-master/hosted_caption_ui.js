/* Administrator caption runs use the same prompt pipeline on a frozen dataset. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
    const button = node("button", "Caption jobs"); button.id = "hostedCaptionJobs"; button.hidden = true;
    document.querySelector(".hosted-actions").append(button);
    const dialog = node("dialog"), form = node("section"), history = node("section"), detail = node("section"), notice = node("p");
    dialog.id = "hostedCaptionDialog"; notice.setAttribute("role", "status");
    const close = node("button", "Close"); close.onclick = () => dialog.close();
    dialog.append(node("h2", "Dataset captions"), node("p", "Generate captions from confirmed images and their approved boxes. A pilot checks a sample before a larger run. Results stay as proposals until you choose to use them. Local dataset runs pause annotation assistance; manual editing stays available."), notice, form, history, detail, close);
    document.body.append(dialog);
    const exportButton = node("button", "Export confirmed captions for training");
    exportButton.type = "button";
    exportButton.onclick = () => { dialog.close(); window.dispatchEvent(new Event("tator:caption-export")); };
    dialog.insertBefore(exportButton, close);
    let epoch = 0, datasetId = "", selected = null, timer = null;
    const valid = (ticket) => ticket === epoch && dialog.open && app.datasetId === datasetId && app.session?.user.role === "admin";
    const api = (path, body) => app.api(path, body === undefined ? {} : {method: "POST", body});
    function action(parent, title, callback) {
        const ticket = epoch; const b = node("button", title); b.type = "button";
        b.onclick = async () => { if (!valid(ticket)) return; b.disabled = true; try { await callback(); } catch (e) { if (valid(ticket) && !e.stale) notice.textContent = e.message; } finally { b.disabled = false; } };
        parent.append(b); return b;
    }
    function field(parent, title, input) { const label = node("label", title); label.append(input); parent.append(label); return input; }
    function textInput(value, type = "text") { const e = node("input"); e.type = type; e.value = value; return e; }
    async function refresh(ticket) {
        const rows = await api("/tools/caption-batches");
        if (!valid(ticket)) return;
        history.replaceChildren(node("h3", "Saved runs"));
        const jobs = rows.filter(row => row.request.dataset_id === datasetId);
        if (!jobs.length) history.append(node("p", "No caption runs for this dataset yet."));
        for (const job of jobs) {
            const row = node("div"); row.className = "hosted-analysis-job";
            row.append(node("span", `${job.request.pilot ? "Pilot" : "Caption run"} · ${job.status} · ${new Date(job.created_at * 1000).toLocaleString()}`));
            action(row, "Inspect", () => show(job.job_id, ticket)); history.append(row);
        }
    }
    async function show(id, ticket) {
        clearTimeout(timer); selected = id;
        const job = await api(`/tools/caption-batches/${id}`);
        if (!valid(ticket) || selected !== id) return;
        detail.replaceChildren(node("h3", `Caption run · ${job.status.replaceAll("_", " ")}`));
        detail.append(node("p", `${job.result.processed || 0} of ${job.result.total_cases || 0} images processed. ${job.result.failed || 0} failed. Existing annotations have not been replaced.${job.error ? " " + job.error : ""}`));
        const certification = job.result.certification;
        if (certification) {
            detail.append(node("strong", certification.status === "certified" ? "Pilot passed its checks" : "Pilot has not passed all checks"));
            const list = node("ul");
            for (const check of certification.checks) list.append(node("li", `${check.passed ? "Passed" : "Needs attention"}: ${check.name.replaceAll("_", " ")}`));
            detail.append(list);
            if (certification.status === "certified") action(detail, "Run this recipe on the full saved snapshot", async () => {
                const next = await api("/tools/caption-batches", {...job.request, operation_id: crypto.randomUUID(), sample_size: 0, pilot: null, certified_pilot_id: id});
                await refresh(ticket); await show(next.job_id, ticket);
            });
        }
        const stopped = ["completed", "failed", "cancelled", "interrupted", "waiting_for_admin"].includes(job.status);
        if (!stopped) {
            action(detail, "Cancel run", async () => { await api(`/tools/caption-batches/${id}/cancel`, {operation_id: crypto.randomUUID()}); await show(id, ticket); });
            timer = setTimeout(() => show(id, ticket).catch(e => { if (valid(ticket)) notice.textContent = e.message; }), 1500);
        } else if (job.status !== "completed") action(detail, "Resume retained work", async () => { await api(`/tools/caption-batches/${id}/resume`, {operation_id: crypto.randomUUID()}); await show(id, ticket); });
        const gallery = node("div"); gallery.className = "hosted-caption-proposals"; detail.append(gallery);
        let offset = 0;
        async function page() {
            const result = await api(`/tools/caption-batches/${id}/captions?offset=${offset}&limit=12`);
            if (!valid(ticket) || selected !== id) return;
            gallery.querySelector("button")?.remove();
            for (const item of result.captions) {
                const card = node("article"), image = node("img"); image.src = `/api/hosted/images/${item.image_id}/media`; image.alt = item.row.image_name; image.loading = "lazy";
                const inspect = node("a", "Review on image"); inspect.href = `/annotate?image=${encodeURIComponent(item.image_id)}&caption_batch=${encodeURIComponent(id)}`;
                card.append(image, node("strong", item.row.image_name), node("p", item.caption || "No accepted caption for this attempt."), inspect); gallery.append(card);
            }
            offset += result.captions.length;
            if (offset < result.total) action(gallery, "Load more captions", page);
        }
        await page();
    }
    async function open() {
        clearTimeout(timer); const ticket = ++epoch; datasetId = app.datasetId; selected = null;
        notice.textContent = ""; form.replaceChildren(); history.replaceChildren(); detail.replaceChildren(); dialog.showModal();
        if (!datasetId) { notice.textContent = "Choose a dataset first."; return; }
        const catalog = await api(`/tools/catalog?dataset_id=${datasetId}`);
        if (!valid(ticket)) return;
        const defaults = catalog.caption_batch_defaults;
        if (!defaults) { notice.textContent = "The server has not enabled dataset caption runs."; return; }
        const provider = node("select"); provider.add(new Option("Local Qwen", "local_qwen"));
        if (catalog.paid_caption?.models.length) provider.add(new Option("OpenAI · metered calls", "openai"));
        field(form, "Caption provider", provider);
        const model = field(form, "Caption model", node("select")), paid = node("section"); form.append(paid);
        paid.append(node("p", "This runs the same caption stages through the configured provider. Each stage is billed; the server checks its spending limits before sending it. Local sampling settings do not apply."));
        const detailChoice = field(paid, "Image detail", node("select"));
        const effort = field(paid, "Reasoning effort", node("select"));
        for (const value of ["", "none", "minimal", "low", "medium", "high", "xhigh"]) effort.add(new Option(value || "Model default", value));
        const limit = field(paid, "Maximum output tokens per stage", textInput(2048, "number")); limit.min = "16";
        function modelChanged() {
            const chosen = catalog.paid_caption?.models.find(item => item.id === model.value);
            if (!chosen) return;
            const previous = detailChoice.value; detailChoice.replaceChildren(...chosen.image_details.map(value => new Option(value, value)));
            detailChoice.value = chosen.image_details.includes(previous) ? previous : chosen.image_details.includes("high") ? "high" : chosen.image_details[0];
            limit.max = String(chosen.max_output_tokens); if (Number(limit.value) > chosen.max_output_tokens) limit.value = chosen.max_output_tokens;
        }
        function providerChanged() {
            model.replaceChildren(); paid.hidden = provider.value !== "openai";
            if (paid.hidden) { model.add(new Option("Server default", "active")); for (const id of catalog.qwen?.allowed_model_ids || []) model.add(new Option(id, id)); }
            else for (const item of catalog.paid_caption.models) model.add(new Option(item.id, item.id));
            modelChanged();
        }
        provider.onchange = providerChanged; model.onchange = modelChanged; providerChanged();
        const prompt = node("textarea"); prompt.rows = 3; prompt.value = defaults.caption_request.user_prompt || ""; field(form, "Caption instructions", prompt);
        const count = field(form, "Pilot sample size", textInput(defaults.sample_size, "number")); count.min = "1";
        const advanced = node("details"); advanced.append(node("summary", "Prompt stages, retry settings and pilot limits")); form.append(advanced);
        advanced.append(node("p", "These are the server's exact caption-pipeline defaults. Edit only the options you want to change. The model and instructions above take precedence."));
        const recipes = {};
        for (const key of ["caption_request", "settings", "pilot"]) {
            const input = node("textarea"); input.rows = 8; input.value = JSON.stringify(defaults[key], null, 2); recipes[key] = field(advanced, key.replaceAll("_", " "), input);
        }
        async function start(pilot) {
            const settings = Object.fromEntries(Object.entries(recipes).map(([key, input]) => [key, JSON.parse(input.value)]));
            settings.caption_request.model_id = model.value; settings.caption_request.user_prompt = prompt.value;
            const providerFields = {provider: provider.value};
            if (provider.value === "openai") {
                if (!limit.reportValidity()) return;
                providerFields.provider_options = {model: model.value, detail: detailChoice.value, reasoning_effort: effort.value || null, max_output_tokens: Number(limit.value), service_tier: "standard"};
                for (const key of ["temperature", "top_p", "top_k", "presence_penalty"]) delete settings.caption_request[key];
                settings.caption_request.max_new_tokens = Number(limit.value);
                settings.caption_request.refinement_model_id = model.value; settings.caption_request.caption_fallback_model_id = model.value;
            }
            if (!pilot && !confirm("Run on all confirmed images without a passing pilot? Results remain proposals. Paid runs incur provider charges; local runs use server compute.")) return;
            const snapshot = await api(`/datasets/${datasetId}/snapshots`, {operation_id: crypto.randomUUID(), name: "Caption run"});
            if (!valid(ticket)) return;
            const job = await api("/tools/caption-batches", {dataset_id: datasetId, snapshot_id: snapshot.id || snapshot.snapshot_id,
                operation_id: crypto.randomUUID(), ...settings, ...providerFields, pilot: pilot ? settings.pilot : null, sample_size: pilot ? Number(count.value) : 0, sample_seed: defaults.sample_seed});
            if (!valid(ticket)) return;
            await refresh(ticket); await show(job.job_id, ticket);
        }
        action(form, "Run pilot", () => start(true));
        action(advanced, "Run full dataset without a pilot…", () => start(false));
        await refresh(ticket);
    }
    button.onclick = () => { const ticket = epoch + 1; open().catch(e => { if (valid(ticket) && !e.stale) notice.textContent = e.message; }); };
    dialog.addEventListener("close", () => { ++epoch; clearTimeout(timer); });
    window.addEventListener("tator:hosted-session", () => { button.hidden = app.session?.user.role !== "admin"; if (button.hidden) { ++epoch; clearTimeout(timer); dialog.close(); form.replaceChildren(); history.replaceChildren(); detail.replaceChildren(); } });
})();
