/* Administrator billing configuration. Credentials are never loaded back. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const node = (tag, text) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; return element; };
    const button = node("button", "Paid services"); button.type = "button"; button.hidden = true; button.id = "hostedProviders"; document.querySelector(".hosted-actions").append(button);
    const dialog = node("dialog"); dialog.id = "hostedProvidersDialog"; document.body.append(dialog);
    let epoch = 0;
    const dollars = (value) => `$${(value / 1000000).toFixed(2)}`;
    async function open() {
        const generation = ++epoch;
        dialog.replaceChildren(node("h2", "Paid caption services")); dialog.showModal();
        const [result, datasets, activity] = await Promise.all([app.api("/providers"), app.api("/datasets"), app.api("/providers/calls?unresolved_only=true&limit=100")]);
        if (generation !== epoch || !dialog.open) return;
        const current = result.providers.find((item) => item.name === "openai") || { revision: 0, enabled: false, monthly_limit: 0, request_limit: 0, allowed_datasets: [], models: [] };
        dialog.append(node("p", "Local Qwen runs on your server. OpenAI uses a paid account. Keep its key on the server and choose which datasets and models may use it."));
        dialog.append(node("p", `This UTC month: ${dollars(current.spent_or_reserved || 0)} used or reserved. ${current.unresolved_calls || 0} calls still need confirmed billing information.`));
        const notice = node("p"); notice.dataset.dialogNotice = ""; notice.hidden = true; notice.setAttribute("role", "status"); dialog.append(notice);
        const form = node("form"); dialog.append(form);
        function input(label, type, value) { const wrap = node("label", label), field = node("input"); field.type = type; field.value = value; wrap.append(field); form.append(wrap); return field; }
        const enabled = input("Enable paid calls", "checkbox", ""); enabled.checked = current.enabled;
        const key = input(current.has_credential ? "Replace API key (leave blank to keep it)" : "OpenAI API key", "password", ""); key.autocomplete = "new-password";
        const monthly = input("Monthly budget (USD)", "number", current.monthly_limit / 1000000); monthly.min = 0; monthly.step = ".01"; monthly.required = true;
        const perCall = input("Maximum reservation per call (USD)", "number", current.request_limit / 1000000); perCall.min = 0; perCall.step = ".01"; perCall.required = true;
        const models = input("Allowed model IDs, separated by commas", "text", current.models.join(", "));
        const group = node("fieldset"); group.append(node("legend", "Allowed datasets"));
        const choices = [];
        for (const dataset of datasets.datasets) { const label = node("label"), check = node("input"); check.type = "checkbox"; check.value = dataset.id; check.checked = current.allowed_datasets.includes(dataset.id); choices.push(check); label.append(check, document.createTextNode(dataset.name)); group.append(label); }
        form.append(group, node("p", "A reservation is an estimated maximum, not a billing guarantee from the provider. An unknown reply keeps its reservation. An observed charge above its estimate pauses paid calls until you review the settings."));
        const save = node("button", "Save paid-service settings"); save.type = "submit"; form.append(save);
        form.onsubmit = (event) => { event.preventDefault(); app.action(save, async () => {
            const body = { operation_id: crypto.randomUUID(), expected_revision: current.revision, enabled: enabled.checked,
                monthly_limit: Math.round(Number(monthly.value) * 1000000), request_limit: Math.round(Number(perCall.value) * 1000000),
                models: models.value.split(",").map((item) => item.trim()).filter(Boolean), allowed_datasets: choices.filter((item) => item.checked).map((item) => item.value) };
            if (key.value) body.credential = key.value;
            try { await app.api("/providers/openai", { method: "PUT", body }); }
            finally { key.value = ""; }
            await open(); app.notice("Paid-service settings saved.");
        }); };
        const billing = node("section"); billing.id = "hostedProviderBilling";
        dialog.append(billing);
        let offset = 0;
        const more = node("button", "Load more unresolved calls"); more.type = "button"; more.id = "hostedProviderBillingMore";
        function renderCalls(calls) {
            if (!offset && calls.length) billing.append(node("h3", "Unresolved calls"), node("p", "Check the provider's usage before recording a bill. Batch billing is available after the provider confirms the batch has finished."));
            for (const call of calls) {
                const row = node("div"), reconcile = node("button", "Record confirmed charge"); reconcile.type = "button";
                row.dataset.providerCall = call.id; row.dataset.billingState = call.state;
                reconcile.disabled = !call.can_reconcile;
                const explanation = call.reason === "batch_terminal_not_verified"
                    ? "Confirm this batch's final status before recording its bill."
                    : call.reason === "batch_usage_unresolved" ? "The batch has finished; its bill still needs confirmation."
                    : call.state === "dispatched" ? "This request is still active." : "Confirm the charge using the provider's records.";
                row.append(node("p", `${call.model} · ${new Date(call.created_at * 1000).toLocaleString()} · ${dollars(call.reserved)} reserved`), node("p", explanation), reconcile);
                reconcile.onclick = () => app.action(reconcile, async () => {
                    const amount = prompt("Confirmed provider charge in USD (0 if the request was never sent):"); if (amount === null) return;
                    if (!/^\d+(\.\d{1,6})?$/.test(amount)) throw new Error("Enter a nonnegative dollar amount, with up to six decimals.");
                    const note = prompt("How did you confirm this charge?"); if (!note) return;
                    try {
                        await app.api(`/providers/calls/${call.id}/reconcile`, { method: "POST", body: { operation_id: crypto.randomUUID(), charged: Math.round(Number(amount) * 1000000), note } });
                    } catch (error) {
                        if (error.status !== 409) throw error;
                        await open(); app.notice("Billing changed while you were confirming it. The current state is shown."); return;
                    }
                    await open();
                }); billing.append(row);
            }
            offset += calls.length;
            more.hidden = calls.length < 100;
        }
        renderCalls(activity.calls);
        more.onclick = () => app.action(more, async () => {
            const next = await app.api(`/providers/calls?unresolved_only=true&limit=100&offset=${offset}`);
            if (generation === epoch && dialog.open) renderCalls(next.calls);
        }); dialog.append(more);
        const close = node("button", "Close"); close.type = "button"; close.onclick = () => dialog.close(); dialog.append(close);
    }
    dialog.addEventListener("close", () => { epoch++; dialog.querySelectorAll('input[type="password"]').forEach((field) => { field.value = ""; }); });
    button.onclick = () => app.action(button, open);
    const session = () => { epoch++; button.hidden = app.session?.user.role !== "admin"; dialog.close(); dialog.replaceChildren(); };
    window.addEventListener("tator:hosted-session", session); session();
})();
