/* Named administrator destinations; hosted operations retain their own authority checks. */
(() => {
    "use strict";
    const canvas = window.TatorHosted, workspace = window.TatorHostedDashboard;
    if (!canvas && !workspace) return;
    const user = () => canvas ? canvas.session?.user : workspace.session?.user;
    const datasetId = () => canvas ? canvas.datasetId : workspace.datasetId;
    const operations = [
        ["dqe", "Data Quality Explorer"], ["pruning", "Dataset pruning"],
        ["ingestion", "Data ingestion"],
    ];
    const training = [
        ["clip_training", "Class predictor"], ["yolo_training", "YOLO"],
        ["sam3_training", "SAM3"], ["rfdetr_training", "RF-DETR"],
        ["qwen_training", "Qwen"], ["salad_training", "SALAD descriptor"],
    ];
    let nav, openedLink = false;
    function button(parent, label, callback) {
        const item = document.createElement("button"); item.type = "button"; item.textContent = label;
        item.onclick = async () => {
            item.disabled = true;
            try { await callback(); }
            catch (error) { if (!error.stale) workspace?.notice(error.message, true); }
            finally { item.disabled = false; }
        };
        parent.append(item); return item;
    }
    async function open(destination) {
        if (user()?.role !== "admin") return;
        if (destination === "models") { document.getElementById("hostedTools")?.click(); return; }
        if (canvas) {
            const query = new URLSearchParams({dataset: datasetId()});
            if (destination !== "datasets") query.set("tool", destination);
            await canvas.leaveFor(`/?${query}`); return;
        }
        if (destination === "profiles") { document.getElementById("hostedProfiles")?.click(); return; }
        if (destination === "datasets") { document.querySelector("#workspace")?.scrollIntoView({block: "start"}); return; }
        await window.TatorHostedAnalysis.open(destination);
    }
    function mount() {
        if (!nav) {
            nav = document.createElement("nav"); nav.id = "hostedAdminNav"; nav.setAttribute("aria-label", "Administrator tools");
            if (canvas) {
                const current = button(nav, "Annotate", () => {}); current.setAttribute("aria-current", "page");
            }
            button(nav, "Datasets", () => open("datasets"));
            for (const [operation, label] of operations) button(nav, label, () => open(operation));
            const menu = document.createElement("details"), summary = document.createElement("summary"), list = document.createElement("div");
            summary.textContent = "Train models"; list.className = "hosted-training-menu";
            const placeMenu = () => {
                if (!menu.open) return;
                list.style.left = "0px";
                const overflow = list.getBoundingClientRect().right - window.innerWidth + 14;
                if (overflow > 0) list.style.left = `${-overflow}px`;
            };
            menu.addEventListener("toggle", placeMenu); window.addEventListener("resize", placeMenu);
            for (const [operation, label] of training) button(list, label, async () => { menu.open = false; await open(operation); });
            menu.append(summary, list); nav.append(menu);
            button(nav, "Tools & models", () => open("models"));
            button(nav, "Analysis presets", () => open("profiles"));
            const anchor = canvas ? document.querySelector(".tab-bar") : document.querySelector(".hosted-header");
            anchor?.insertAdjacentElement("afterend", nav);
        }
        nav.hidden = user()?.role !== "admin";
    }
    window.addEventListener("tator:hosted-canvas", mount);
    window.addEventListener("tator:hosted-session", mount);
    window.addEventListener("tator:hosted-workspace", async () => {
        mount();
        const requested = new URLSearchParams(location.search).get("tool");
        const known = ["profiles", "models", "caption_export", ...operations.map(item => item[0]), ...training.map(item => item[0])];
        if (openedLink || !known.includes(requested) || user()?.role !== "admin" || !datasetId()) return;
        openedLink = true;
        try { await open(requested); }
        catch (error) { if (!error.stale) workspace.notice(error.message, true); }
    });
    if (canvas) window.addEventListener("storage", event => { if (event.key === "tator.hosted.session-event" && nav) nav.hidden = true; });
    mount();
})();
