/* Dataset model settings and private tool status; shared by workspace and canvas. */
(() => {
    "use strict";
    const node = (tag, text) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; return element; };
    let dialog, content, notice, button, stateLabel, epoch = 0, pendingPoll = false;
    const canvas = () => window.TatorHosted;
    const dashboard = () => window.TatorHostedDashboard;
    const datasetId = () => canvas()?.datasetId || dashboard()?.datasetId;
    const user = () => window.TatorHostedBoot?.user || dashboard()?.session?.user;
    function api(path, body) {
        if (canvas()) return canvas().api(`/api/hosted${path}`, body);
        return dashboard().api(path, body === undefined ? {} : { method: "POST", body });
    }
    function errorText(error) { notice.textContent = error.message; notice.hidden = false; }
    function action(parent, label, callback) {
        const control = node("button", label); control.type = "button";
        const current = epoch, id = datasetId(), session = dashboard()?.session;
        const valid = () => current === epoch && id === datasetId() && session === dashboard()?.session && dialog.open && control.isConnected;
        control.onclick = async () => {
            if (!valid() || control.disabled) return;
            control.disabled = true; notice.hidden = true;
            try { await callback(valid); }
            catch (error) { if (valid() && !error.stale) errorText(error); }
            finally { control.disabled = false; }
        };
        parent.append(control); return control;
    }
    function section(title, explanation) {
        const result = node("section"); result.append(node("h3", title), node("p", explanation)); content.append(result); return result;
    }
    function choices(parent, label, models, selected) {
        const wrap = node("label", label), select = node("select");
        select.setAttribute("aria-label", label);
        const empty = node("option", "Choose a model"); empty.value = ""; select.append(empty);
        for (const model of models) { const option = node("option", model.model_id); option.value = model.model_id; select.append(option); }
        select.value = selected || ""; wrap.append(select); parent.append(wrap); return select;
    }
    async function render() {
        const generation = ++epoch, id = datasetId();
        content.replaceChildren(); notice.hidden = true;
        if (!id) { content.append(node("p", "Choose a dataset first.")); return; }
        try {
            const [catalog, dataset] = await Promise.all([api(`/tools/catalog?dataset_id=${encodeURIComponent(id)}`), api(`/datasets/${encodeURIComponent(id)}`)]);
            if (generation !== epoch || datasetId() !== id || !dialog.open) return;
            const sam = section("Your segmentation model", "This model and its image cache belong to your account. Selecting it does not change another person's tools.");
            const select = choices(sam, "SAM model", catalog.models.filter((model) => ["sam1", "sam3"].includes(model.kind) && catalog.sam_allowed_model_ids?.includes(model.model_id)), catalog.sam?.model_id);
            action(sam, "Use this SAM model", async (valid) => {
                if (!select.value) throw new Error("Choose an installed model.");
                await api("/tools/sam", { model_id: select.value, expected_revision: catalog.sam?.revision || 0, operation_id: crypto.randomUUID() });
                if (!valid()) return; await render(); await poll();
            });
            for (const [kind, title, explanation] of [
                ["detector", "Dataset detector", "Everyone working on this dataset uses this detector. Requests run one at a time on its shared instance."],
                ["classifier", "Dataset class predictor", "Each person's tools load their own copy of this trained class predictor. Its labels are linked to the dataset's classes below."],
                ["edr", "Dataset detection recipe", "Run the dataset's saved detection recipe, including its trained ensemble and selected model stages. Requests share the dataset's inference queue."],
            ]) {
                const panel = section(title, explanation);
                panel.append(node("p", `Current model: ${catalog[kind]?.model_id || "Not selected"}`));
                if (user()?.role !== "admin") continue;
                const models = catalog.models.filter((model) => kind === "detector" ? ["yolo", "rfdetr"].includes(model.kind) : model.kind === kind);
                const choice = choices(panel, `${title} model`, models, catalog[kind]?.model_id), mapping = node("div"); mapping.className = "hosted-model-mapping"; panel.append(mapping);
                function drawMapping() {
                    mapping.replaceChildren();
                    const model = models.find((item) => item.model_id === choice.value);
                    if (!model) return;
                    for (const label of model.labels || []) {
                        const row = node("label", `${label} → `), select = node("select"); select.dataset.modelLabel = label;
                        select.setAttribute("aria-label", `Dataset class for ${label}`);
                        const empty = node("option", "Choose dataset class"); empty.value = ""; select.append(empty);
                        for (const cls of dataset.classes.filter((cls) => !cls.retired)) { const option = node("option", cls.name); option.value = cls.id; select.append(option); }
                        select.value = (catalog[kind]?.model_id === choice.value && catalog[kind]?.class_mapping?.[label]) || dataset.classes.find((cls) => !cls.retired && cls.name === label)?.id || "";
                        row.append(select); mapping.append(row);
                    }
                }
                choice.onchange = drawMapping; drawMapping();
                action(panel, `Set dataset ${kind}`, async (valid) => {
                    if (!choice.value) throw new Error("Choose an installed model.");
                    const pairs = [...mapping.querySelectorAll("select")].map((item) => [item.dataset.modelLabel, item.value]);
                    if (pairs.some(([, value]) => !value)) throw new Error("Link every model label to a dataset class.");
                    await api(`/tools/datasets/${encodeURIComponent(id)}/${kind}`, { model_id: choice.value, expected_revision: catalog.dataset_revision, class_mapping: Object.fromEntries(pairs), operation_id: crypto.randomUUID() });
                    if (!valid()) return; await render(); await poll();
                });
            }
            const recipes = section("Saved SAM3 recipes", "An administrator attaches installed recipes to this dataset. You can combine them into your own cascade presets in the annotation pane. Each recipe keeps its saved model and classifier settings.");
            const attached = new Set(catalog.sam3_recipes?.model_ids || []);
            recipes.append(node("p", attached.size ? `Available recipes: ${[...attached].join(", ")}` : "No saved SAM3 recipes are attached yet."));
            if (user()?.role === "admin") {
                const list = node("div"); recipes.append(list);
                for (const model of catalog.models.filter((item) => item.kind === "sam3_recipe")) {
                    const label = node("label"), check = node("input"); check.type = "checkbox"; check.value = model.model_id; check.checked = attached.has(model.model_id);
                    label.append(check, document.createTextNode(model.model_id)); list.append(label);
                }
                action(recipes, "Save available recipes", async (valid) => {
                    await api(`/tools/datasets/${encodeURIComponent(id)}/sam3-recipes`, { model_ids: [...list.querySelectorAll("input:checked")].map((item) => item.value),
                        expected_revision: catalog.dataset_revision, operation_id: crypto.randomUUID() });
                    if (!valid()) return; await render(); await poll();
                });
            }
            const qwen = section("Local Qwen captioning and detection", "Your requests use a private worker. Choose among installed models in the caption controls; the server defines the default and permitted models.");
            qwen.append(node("p", `Default: ${catalog.qwen?.default_model_id || "Not configured"}`));
            const qwenModels = catalog.models.filter((model) => model.kind === "qwen" && catalog.qwen?.allowed_model_ids?.includes(model.model_id));
            qwen.append(node("p", qwenModels.length ? `Installed choices: ${qwenModels.map((model) => model.model_id).join(", ")}` : "No permitted Qwen weights have been installed yet."));
            if (user()?.role === "admin") {
                const install = section("Install configured models", "Model files are configured on the server. Installation copies their exact weights into recoverable storage.");
                const choicesList = (catalog.available_to_install || []).map((model_id) => ({ model_id }));
                const choice = choices(install, "Server model", choicesList);
                action(install, "Install model", async (valid) => {
                    if (!choice.value) throw new Error("Choose a configured server model.");
                    await api(`/tools/models/${encodeURIComponent(choice.value)}/install`, { operation_id: crypto.randomUUID() }); if (!valid()) return; await render(); await poll();
                });
                if (!choicesList.length) install.append(node("p", "No server model files have been configured yet."));
            }
        } catch (error) { if (generation === epoch && datasetId() === id && dialog.open && !error.stale) errorText(error); }
    }
    async function poll() {
        const id = datasetId(), current = epoch;
        if (!id || !user() || pendingPoll || document.hidden) return;
        pendingPoll = true;
        try {
            const value = await api(`/tools/state?dataset_id=${encodeURIComponent(id)}`);
            if (current !== epoch || datasetId() !== id) return;
            const phase = { pausing: "Pausing assistance for administrator compute", draining: "Finishing active tools before administrator compute", unloading: "Unloading assistance for administrator compute", heavy: "Assistance paused for administrator compute", restoring: "Reloading your annotation tools", failed: "Assistance recovery failed; an administrator needs to check the runtime", closed: "Annotation assistance is offline" }[value.phase];
            stateLabel.textContent = phase ? `${phase}. You can still annotate and save manually.` : ({ loading: "Loading your annotation tools…", busy: "Your annotation tool is working…", unloaded: "Tools load when you first use them", ready: "Annotation tools ready", failed: "A tool failed to load. Open Tools for its settings." }[value.user.state] || "Annotation tools available");
            const configured = Object.values(value.capabilities || {}).some((item) => item.configured);
            if (!configured && value.phase === "available") stateLabel.textContent = "No assistance models configured · manual annotation is available";
            if (canvas()) {
                canvas().setToolsState(value);
                const catalog = await api(`/tools/catalog?dataset_id=${encodeURIComponent(id)}`);
                if (current === epoch && datasetId() === id) window.TatorHostedCanvas?.configureTools(catalog, value);
            }
        } catch (error) { if (current === epoch && datasetId() === id && !error.stale) stateLabel.textContent = `Tool status unavailable: ${error.message}`; }
        finally { pendingPoll = false; }
    }
    function mount() {
        if (button) { button.hidden = !user(); return; }
        const parent = document.querySelector(".hosted-canvas-feedback") || document.querySelector(".hosted-actions");
        if (!parent) return;
        button = node("button", "Tools & models"); button.type = "button"; button.id = "hostedTools";
        stateLabel = node("span"); stateLabel.id = "hostedToolsState"; stateLabel.setAttribute("role", "status"); parent.append(button, stateLabel);
        dialog = node("dialog"); dialog.id = "hostedToolsDialog"; dialog.append(node("h2", "Annotation tools"));
        notice = node("p"); notice.setAttribute("role", "alert"); notice.hidden = true;
        content = node("div"); dialog.append(notice, content);
        const close = node("button", "Close"); close.type = "button"; close.onclick = () => dialog.close(); dialog.append(close); document.body.append(dialog);
        dialog.addEventListener("close", () => { epoch++; });
        button.onclick = () => { dialog.showModal(); render(); };
        button.hidden = !user(); poll();
    }
    window.addEventListener("tator:hosted-canvas", mount);
    window.addEventListener("tator:hosted-session", () => { epoch++; dialog?.close(); content?.replaceChildren(); if (stateLabel) stateLabel.textContent = ""; mount(); });
    window.addEventListener("DOMContentLoaded", mount);
    setInterval(poll, 5000);
})();
