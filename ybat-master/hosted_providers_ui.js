/* Paid caption services are not offered in the workspace yet. The provider API remains server-side. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const node = (tag, text) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; return element; };
    const button = node("button", "Paid services"); button.type = "button"; button.hidden = true; button.id = "hostedProviders"; document.querySelector(".hosted-actions").append(button);
    const dialog = node("dialog"); dialog.id = "hostedProvidersDialog"; document.body.append(dialog);
    function open() {
        const close = node("button", "Close"); close.type = "button"; close.onclick = () => dialog.close();
        const actions = node("div"); actions.className = "dialog-actions"; actions.append(close);
        dialog.replaceChildren(node("h2", "Paid services"),
            node("p", "Paid caption services, such as OpenAI, are coming later. Annotation tools and the models installed on this server work without them."), actions);
        dialog.showModal(); close.focus();
    }
    button.onclick = open;
    // Hidden until a paid provider is actually offered here; the dialog only says "coming later".
    const offered = false;
    const session = () => { button.hidden = !offered || app.session?.user.role !== "admin"; if (dialog.open) dialog.close(); };
    window.addEventListener("tator:hosted-session", session); session();
})();
