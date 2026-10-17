/* Dataset-associated SALAD heads retain their original trained artifacts. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
    const button = node("button", "Analysis presets"); button.id = "hostedProfiles"; button.hidden = true;
    document.querySelector(".hosted-actions").append(button);
    const dialog = node("dialog"), notice = node("p"), catalog = node("section"), list = node("section"), recipes = node("details"), sources = node("section");
    dialog.id = "hostedProfilesDialog"; notice.setAttribute("role", "status");
    const close = node("button", "Close"); close.onclick = () => dialog.close();
    dialog.append(node("h2", "Analysis presets & trained descriptors"), node("p", "Use the established Data Quality Explorer presets across datasets. Keep trained SALAD descriptors attached to the dataset they describe; completed analyses retain their original recipe."), notice, catalog, list, recipes, sources, close); document.body.append(dialog);
    let epoch = 0, datasetId, snapshot, authority, pending = false;
    const valid = ticket => epoch === ticket && dialog.open && app.datasetId === datasetId && app.session === authority && app.session?.user.role === "admin";
    const api = (path, body) => app.api(`/datasets/${encodeURIComponent(datasetId)}/salad-profiles${path}`, body === undefined ? {} : {method: "POST", body});
    function action(parent, title, callback) {
        const b = node("button", title); b.type = "button";
        b.onclick = async () => {
            const ticket = epoch; if (!valid(ticket) || pending) return;
            pending = true;
            const controls = [...dialog.querySelectorAll("button,input,select,textarea")];
            const disabled = controls.map(control => control.disabled);
            controls.forEach(control => { if (control !== close) control.disabled = true; });
            try { await callback(); }
            catch (e) { if (valid(ticket) && !e.stale) notice.textContent = e.message; }
            finally {
                pending = false; controls.forEach((control, i) => { control.disabled = disabled[i]; });
                for (const panel of recipes.querySelectorAll("[data-recipe-role]")) panel.refreshButtons?.();
            }
        };
        parent.append(b); return b;
    }
    async function mutate(value, ticket) {
        await api("", {...value, expected_revision: snapshot.revision, operation_id: crypto.randomUUID()});
        if (valid(ticket)) { notice.textContent = "Dataset recipe settings saved. Run a new analysis to use them."; await render(ticket); }
    }
    function label(text, control) { const l = node("label", text); l.append(control); return l; }
    async function renderCatalog(ticket) {
        const value = await app.api("/analysis-presets"); if (!valid(ticket)) return;
        catalog.replaceChildren(node("h3", "Shared object-analysis presets"), node("p", "Choose a preset when starting Data Quality Explorer. Thorough is the established default: three image encoders plus a trained SALAD aggregator. These are shared recipes, so each dataset does not need its own manual setup."));
        for (const preset of value.presets) {
            const row = node("div"); row.className = "hosted-profile-row";
            row.append(node("strong", preset.label + (preset.id === value.default_recipe ? " · default" : "")), node("p", preset.components.map(c => `${c.label} ${(100 * c.weight).toFixed(3).replace(/\.?0+$/, "")}%`).join(" + ")));
            catalog.append(row);
        }
        catalog.append(node("p", "Each component is normalized before mixing. The established size-and-shape correction is fitted on the analysed dataset for DINOv3, SAM3 and crop SALAD; C-RADIO keeps its original features. Alternative C-RADIO corrections remain experiments, not a new default."), node("p", "Whole-image SALAD uses its own trained heads. Pruning then combines picture similarity with object similarity; that final balance is separate from the backbone weights above."));
        const open = node("button", "Open Data Quality Explorer"); open.type = "button";
        open.disabled = !datasetId; open.onclick = () => { dialog.close(); window.TatorHostedAnalysis?.open("dqe"); }; catalog.append(open);
    }
    async function render(ticket) {
        const value = await api(""); if (!valid(ticket)) return; snapshot = value;
        list.replaceChildren(node("h3", "Trained descriptors for this dataset"), node("p", "A SALAD head contains learned weights, not the dataset images. Its encoder, training scope and exact weight file travel with the association. Attaching a head preserves that training history; it does not retrain it.")); recipes.replaceChildren(node("summary", "Advanced: dataset-specific pruning overrides"), node("p", "Component weights and preprocessing below affect pruning. Data Quality Explorer keeps its selected preset and reuses the chosen compatible crop SALAD head. An incompatible head needs a matching recipe; it is never silently retrained. Existing results keep their captured settings."));
        if (!value.heads.length) list.append(node("p", "No trained SALAD descriptors have been registered yet. Discover retained training outputs below, or import an existing dataset with its descriptors."));
        for (const head of value.heads) {
            const row = node("div"); row.className = "hosted-profile-row";
            const display = node("input"); display.value = head.display_label || head.label || head.id; display.maxLength = 160;
            row.append(label(`${head.encoder_type} · ${head.representation_scope === "bbox_crop" ? "Object crops" : "Whole images"}`, display));
            row.append(node("small", `${head.encoder_model} · ${head.descriptor_dim || "—"} dimensions${head.train_image_count ? ` · trained on ${head.train_image_count} ${head.representation_scope === "bbox_crop" ? "crops" : "images"}` : ""} · weights ${head.head_sha256.slice(0, 12)}${head.archived ? " · Archived" : ""}`));
            action(row, "Save name", () => mutate({action: "rename", head_id: head.id, display_label: display.value}, ticket));
            const attached = value.dataset.head_ids.includes(head.id);
            action(row, attached ? "Detach" : "Attach to dataset", () => mutate({action: attached ? "detach" : "attach", head_id: head.id}, ticket));
            action(row, head.archived ? "Restore descriptor" : "Archive descriptor", () => mutate({action: "archive", head_id: head.id, archived: !head.archived}, ticket));
            list.append(row);
        }
        for (const [role, title] of [["whole_image", "Whole-image recipe"], ["bbox_crop", "Object-crop recipe"]]) {
            recipes.append(recipeEditor(value, role, title, ticket));
        }
    }
    function recipeEditor(value, role, title, ticket) {
        const panel = node("section"); panel.dataset.recipeRole = role;
        panel.append(node("h4", title));
        const current = value.dataset.preferred_recipes[role];
        panel.append(node("p", current ? "Saved pruning override. Changes apply to the next pruning analysis." : "No override: pruning uses its analysis defaults. Add components only if you want a dataset-specific recipe."));
        const names = {dinov3: "DINOv3", cradio: "C-RADIO", sam3: "SAM3"};
        const aliases = {dinov3_pooled: "dinov3", cradio_pooled: "cradio", sam3_mask: "sam3"};
        const base = c => aliases[c.encoder_type || c.encoder || c.backbone || c.id] || c.encoder_type || c.encoder || c.backbone || c.id;
        const key = c => c.head_id ? `salad:${c.head_id}` : `base:${base(c)}`;
        const choices = (role === "whole_image" ? ["dinov3", "cradio"] : ["dinov3", "sam3", "cradio"])
            .map(id => ({key: `base:${id}`, label: names[id], component: {encoder_type: id, encoder: role === "whole_image" ? `${id}_pooled` : id, ...(role === "bbox_crop" ? {id} : {})}}));
        for (const head of value.heads.filter(h => value.dataset.head_ids.includes(h.id) && !h.archived && h.representation_scope === role && (role === "whole_image" || h.encoder_type === "dinov3"))) {
            choices.push({key: `salad:${head.id}`, label: `${head.display_label || head.label || head.id} · SALAD on ${names[head.encoder_type] || head.encoder_type}`,
                component: {encoder_type: head.encoder_type, encoder: role === "whole_image" ? "local_salad" : head.encoder_type, ...(role === "bbox_crop" ? {id: "salad"} : {}), encoder_model: head.encoder_model, head_id: head.id, head_sha256: head.head_sha256}});
        }
        let recipe = structuredClone(current || {components: []}), advancedDirty = false;
        const rows = node("div"), shares = node("p"), hint = node("p"), tools = node("div"), advanced = node("details"), editor = node("textarea");
        tools.className = "hosted-actions"; shares.setAttribute("aria-live", "polite");
        hint.textContent = role === "whole_image" ? "Combine DINOv3 and C-RADIO, optionally with attached SALAD descriptors. SAM3 is used for object crops." : "Combine up to three backbones: DINOv3, SAM3 and C-RADIO. Attached SALAD descriptors can contribute alongside them.";
        advanced.append(node("summary", "Advanced: exact recipe and preprocessing")); editor.rows = 8;
        advanced.append(node("p", "All saved options remain here, including model versions, pooling, size correction and preprocessing. Editing a weight above changes only that weight. Use the button below to load JSON changes into the visual editor."), label("Recipe JSON", editor));
        const parse = () => {
            const candidate = JSON.parse(editor.value);
            if (!candidate || Array.isArray(candidate) || typeof candidate !== "object" || !Array.isArray(candidate.components) || !candidate.components.length || candidate.components.length > 12 || candidate.components.some(c => !c || Array.isArray(c) || typeof c !== "object" || !Number.isFinite(c.weight ?? 1) || (c.weight ?? 1) <= 0) || !Number.isFinite(candidate.components.reduce((n, c) => n + (c.weight ?? 1), 0))) throw new Error("Use 1–12 descriptor components with positive numeric weights. The server checks backbone and SALAD compatibility when saved.");
            return candidate;
        };
        const sync = () => { editor.value = JSON.stringify(recipe, null, 2); advancedDirty = false; };
        const updateShares = () => {
            const total = recipe.components.reduce((n, c) => n + Number(c.weight ?? 1), 0);
            shares.textContent = recipe.components.length ? "Relative contribution: " + recipe.components.map(c => `${choices.find(option => option.key === key(c))?.label || base(c)} ${Number.isFinite(total) && total > 0 ? (100 * (c.weight ?? 1) / total).toFixed(1) : "—"}%`).join(" · ") : "No dataset-specific components selected.";
        };
        const loadJSON = () => { recipe = parse(); advancedDirty = false; draw(); };
        editor.addEventListener("input", () => {
            advancedDirty = true;
            for (const control of rows.querySelectorAll("input,select,button")) control.disabled = true;
            add.disabled = true;
            shares.textContent = "JSON has unsaved changes. Load it into the editor, or save it directly below.";
            save.disabled = false;
        });
        action(advanced, "Load JSON into visual editor", loadJSON);
        const add = action(tools, "Add descriptor", () => {
            const available = choices.find(choice => !recipe.components.some(c => key(c) === choice.key));
            if (available) { recipe.components.push({...available.component, weight: 1}); draw(); }
        });
        const save = action(tools, "Save pruning override", () => {
            if (!advancedDirty) {
                for (const input of rows.querySelectorAll('input[type="number"]')) {
                    if (!Number.isFinite(Number(input.value)) || Number(input.value) <= 0) throw new Error("Enter a positive numeric weight for every descriptor.");
                }
            }
            const selected = advancedDirty ? parse() : structuredClone(recipe);
            return mutate({action: "prefer_recipe", role, recipe: selected}, ticket);
        });
        panel.refreshButtons = () => {
            add.disabled = advancedDirty || recipe.components.length >= 12 || choices.every(choice => recipe.components.some(c => key(c) === choice.key));
            save.disabled = !advancedDirty && !recipe.components.length;
        };
        function draw() {
            rows.replaceChildren();
            recipe.components.forEach((component, index) => {
                const row = node("div"); row.className = "hosted-profile-row";
                const select = node("select"); select.setAttribute("aria-label", `${title}: descriptor ${index + 1}`);
                for (const choice of choices) select.add(new Option(choice.label, choice.key));
                const selectedKey = key(component);
                if (!choices.some(choice => choice.key === selectedKey)) select.add(new Option(`Saved component · ${component.head_id || base(component) || "custom"}`, selectedKey));
                select.value = selectedKey;
                const slider = node("input"); slider.type = "range"; slider.min = "0.05"; slider.max = String(Math.max(4, Number(component.weight ?? 1) * 2)); slider.step = "0.05"; slider.value = String(component.weight ?? 1);
                slider.setAttribute("aria-label", `${title}: weight ${index + 1}`);
                const weight = node("input"); weight.type = "number"; weight.min = "0"; weight.step = "any"; weight.value = String(component.weight ?? 1); weight.style.width = "100px";
                weight.setAttribute("aria-label", `${title}: exact weight ${index + 1}`);
                const changeWeight = value => { component.weight = value; sync(); updateShares(); };
                slider.oninput = () => { weight.setCustomValidity(""); weight.value = slider.value; changeWeight(Number(slider.value)); };
                weight.onchange = () => {
                    const number = Number(weight.value);
                    if (!Number.isFinite(number) || number <= 0) { weight.setCustomValidity("Enter a positive weight."); weight.reportValidity(); return; }
                    weight.setCustomValidity(""); slider.max = String(Math.max(4, number * 2)); slider.value = String(number); changeWeight(number);
                };
                select.onchange = () => {
                    const selected = choices.find(choice => choice.key === select.value);
                    if (!selected || selected.key === selectedKey) return;
                    if (recipe.components.some((other, i) => i !== index && key(other) === selected.key)) { select.value = selectedKey; notice.textContent = "That descriptor is already in this recipe. Adjust its weight instead."; return; }
                    const kept = {...component};
                    for (const field of ["encoder_type", "encoder", "backbone", "id", "head_id", "head_sha256", "model", "encoder_model", "pooling", "cradio_pooling", "input_size"]) delete kept[field];
                    recipe.components[index] = {...kept, ...selected.component}; draw();
                };
                row.append(label("Descriptor", select), label("Relative weight", slider), weight);
                const remove = action(row, "Remove", () => { recipe.components.splice(index, 1); draw(); }); remove.disabled = recipe.components.length === 1;
                if (component.pooling || component.model || component.encoder_model) row.append(node("small", [component.model || component.encoder_model, component.pooling].filter(Boolean).join(" · ")));
                if (!choices.some(choice => choice.key === selectedKey)) row.append(node("small", "This saved descriptor is unavailable here. Choose an attached, active descriptor before saving."));
                rows.append(row);
            });
            sync(); updateShares(); panel.refreshButtons();
        }
        panel.append(hint, rows, shares, tools, advanced); draw(); return panel;
    }
    async function discover(ticket, offset = 0) {
        const result = await api(`/sources?offset=${offset}&limit=25`); if (!valid(ticket)) return;
        if (!offset) sources.replaceChildren(node("h3", "Retained training outputs"));
        sources.querySelector('[data-more]')?.remove();
        if (!result.total) sources.append(node("p", "No retained SALAD training outputs were found for this dataset."));
        for (const source of result.sources) {
            const row = node("div"); row.className = "hosted-profile-row";
            row.append(node("strong", source.label || source.name), node("small", `${source.encoder_type || "Unknown encoder"} · ${source.representation_scope || "Unverified"} · ${source.origin}`));
            if (source.status === "available") action(row, "Register and attach", async () => {
                await api("/register", {source_id: source.source_id, display_label: source.label, expected_revision: snapshot.revision, operation_id: crypto.randomUUID()});
                if (valid(ticket)) { notice.textContent = "Descriptor registered and attached. Original trained weights are preserved."; await render(ticket); }
            });
            else row.append(node("p", "This output did not pass the trained-head format check."));
            sources.append(row);
        }
        if (result.next_offset !== null) { const more = action(sources, "More retained outputs", () => discover(ticket, result.next_offset)); more.dataset.more = ""; }
    }
    button.onclick = async () => {
        const ticket = ++epoch; datasetId = app.datasetId; authority = app.session; snapshot = null; notice.textContent = ""; catalog.replaceChildren(); list.replaceChildren(); recipes.replaceChildren(); recipes.open = false; recipes.hidden = !datasetId; sources.replaceChildren(); dialog.showModal();
        try { await renderCatalog(ticket); if (!datasetId) { notice.textContent = "Select a dataset to manage its trained descriptors or pruning overrides."; return; } await render(ticket); if (valid(ticket)) action(sources, "Discover retained SALAD training outputs", () => discover(ticket)); }
        catch (e) { if (valid(ticket) && !e.stale) notice.textContent = e.message; }
    };
    dialog.addEventListener("close", () => { ++epoch; });
    window.addEventListener("tator:hosted-session", () => { button.hidden = app.session?.user.role !== "admin"; if (dialog.open || button.hidden) { ++epoch; dialog.close(); list.replaceChildren(); recipes.replaceChildren(); sources.replaceChildren(); } });
})();
