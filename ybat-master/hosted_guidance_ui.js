/* Administrator-owned dataset language, shared with annotation assistance. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard; if (!app) return;
    const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
    const button = node("button", "Dataset guidance"), dialog = node("dialog"); button.id = "hostedGuidance"; button.hidden = true;
    document.querySelector(".hosted-actions").append(button); document.body.append(dialog);
    let epoch = 0;
    button.onclick = async () => {
        const ticket = ++epoch, datasetId = app.datasetId;
        dialog.replaceChildren(node("h2", "Dataset guidance")); dialog.showModal();
        const message = node("p"); message.setAttribute("role", "status"); dialog.append(message);
        const close = node("button", "Close"); close.onclick = () => dialog.close(); dialog.append(close);
        const valid = () => ticket === epoch && dialog.open && app.datasetId === datasetId && app.session?.user.role === "admin";
        if (!datasetId) { message.textContent = "Choose a dataset first."; return; }
        try {
            const dataset = await app.api(`/datasets/${datasetId}`); if (!valid()) return;
            const form = node("form"); dialog.insertBefore(form, close);
            form.append(node("p", "Describe the dataset and the meaning of its classes. These defaults help annotation tools use consistent language. They do not change annotations or earlier analysis results."));
            function field(title, value, rows) { const label = node("label", title), area = node("textarea"); area.value = value; area.rows = rows; area.setAttribute("aria-label", title); label.append(area); form.append(label); return area; }
            const context = field("Dataset context", dataset.settings.context || "", 4); context.maxLength = 32000;
            const glossary = field("Class glossary", dataset.settings.labelmap_glossary || "", 7); glossary.maxLength = 64000;
            glossary.placeholder = "Car: passenger car, SUV\nBus: passenger bus, coach";
            form.append(node("p", `Use one class per line, followed by a colon and its terms. Current classes: ${dataset.classes.filter(c => !c.retired).map(c => c.name).join(", ")}.`));
            const save = node("button", "Save guidance"); save.type = "submit"; form.append(save);
            form.onsubmit = async event => {
                event.preventDefault(); if (!valid()) return; save.disabled = true;
                try {
                    const saved = await app.api(`/datasets/${datasetId}/guidance`, {method: "POST", body: {context: context.value, labelmap_glossary: glossary.value, expected_revision: dataset.revision, operation_id: crypto.randomUUID()}});
                    if (valid()) { dataset.revision = saved.revision; message.textContent = "Guidance saved for future tool requests and analyses."; }
                } catch (e) { if (valid() && !e.stale) message.textContent = e.message; }
                finally { if (valid()) save.disabled = false; }
            };
        } catch (e) { if (valid() && !e.stale) message.textContent = e.message; }
    };
    dialog.addEventListener("close", () => ++epoch);
    window.addEventListener("tator:hosted-session", () => { button.hidden = app.session?.user.role !== "admin"; if (button.hidden) { ++epoch; dialog.close(); dialog.replaceChildren(); } });
})();
