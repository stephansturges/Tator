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
    // With a workbench service, these open the full legacy tabs instead of the
    // hosted snapshot launcher, which stays available as "Server jobs".
    const workbenchTabs = {dqe: "class-split", pruning: "dataset-pruning", ingestion: "data-ingestion", clip_training: "training",
        yolo_training: "yolo-train", sam3_training: "sam3-train", rfdetr_training: "rfdetr-train", qwen_training: "qwen-train",
        salad_training: "data-ingestion", recipes: "prepass-builder", mining: "agent-mining", vocabulary: "prompt-helper",
        workbench_datasets: "datasets", model_settings: "detectors", workbench_labeling: "labeling"};
    const moreTools = [["recipes", "Detection recipes"], ["mining", "SAM3 recipe mining"], ["vocabulary", "SAM3 vocabulary explorer"],
        ["workbench_labeling", "Workbench annotation"], ["workbench_datasets", "Workbench datasets"], ["model_settings", "Model settings"]];
    const workbench = () => Boolean(canvas ? window.TatorHostedBoot?.workbench : workspace.session?.workbench);
    let nav, toggle, openedLink = false;
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
        if (workbench() && workbenchTabs[destination]) {
            const query = new URLSearchParams({tab: workbenchTabs[destination]});
            if (datasetId()) query.set("dataset", datasetId());
            if (canvas) await canvas.leaveFor(`/workbench?${query}`); else location.assign(`/workbench?${query}`);
            return;
        }
        if (destination === "jobs") {
            if (canvas) { await canvas.leaveFor(`/?${new URLSearchParams({dataset: datasetId(), tool: "jobs"})}`); return; }
            await window.TatorHostedAnalysis.open(null); return;
        }
        if (canvas) {
            const query = new URLSearchParams({dataset: datasetId()});
            if (destination !== "datasets") query.set("tool", destination);
            await canvas.leaveFor(`/?${query}`); return;
        }
        if (destination === "profiles") { document.getElementById("hostedProfiles")?.click(); return; }
        if (destination === "datasets") { document.querySelectorAll("dialog[open]").forEach(dialog => dialog.close()); window.scrollTo({top: 0}); return; }
        await window.TatorHostedAnalysis.open(destination);
    }
    function mount() {
        if (!nav) {
            nav = document.createElement("nav"); nav.id = "hostedAdminNav"; nav.setAttribute("aria-label", "Administrator tools");
            // On a phone the strip folds behind one button so the work starts near the top.
            toggle = document.createElement("button"); toggle.type = "button"; toggle.className = "hosted-nav-toggle"; toggle.textContent = "Administrator tools";
            toggle.setAttribute("aria-expanded", "false"); nav.dataset.collapsed = "true";
            toggle.onclick = () => { const open = nav.dataset.collapsed === "true"; nav.dataset.collapsed = String(!open); toggle.setAttribute("aria-expanded", String(open)); };
            nav.append(toggle);
            if (canvas) {
                const current = button(nav, "Annotate", () => {}); current.setAttribute("aria-current", "page");
            }
            const datasets = button(nav, "Datasets", () => open("datasets"));
            if (!canvas) datasets.setAttribute("aria-current", "page");
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
            const more = document.createElement("details"), moreSummary = document.createElement("summary"), moreList = document.createElement("div");
            more.id = "hostedMoreTools"; moreSummary.textContent = "More tools"; moreList.className = "hosted-training-menu";
            for (const [destination, label] of moreTools) button(moreList, label, async () => { more.open = false; await open(destination); });
            more.append(moreSummary, moreList); nav.append(more);
            const jobs = button(nav, "Server jobs", () => open("jobs")); jobs.id = "hostedServerJobs";
            button(nav, "Tools & models", () => open("models"));
            button(nav, "Analysis presets", () => open("profiles"));
            const anchor = canvas ? document.querySelector(".tab-bar") : document.querySelector(".hosted-header");
            anchor?.insertAdjacentElement("afterend", nav);
        }
        nav.hidden = user()?.role !== "admin";
        const more = nav.querySelector("#hostedMoreTools");
        if (more) more.hidden = !workbench();
    }
    window.addEventListener("tator:hosted-canvas", mount);
    window.addEventListener("tator:hosted-session", mount);
    window.addEventListener("tator:hosted-workspace", async () => {
        mount();
        const requested = new URLSearchParams(location.search).get("tool");
        const known = ["profiles", "models", "caption_export", "jobs", ...operations.map(item => item[0]), ...training.map(item => item[0])];
        if (openedLink || !requested) return;
        // The link is handled once; reloading the page afterwards does not reopen it.
        workspace.dropRouteParams?.("tool");
        if (!known.includes(requested) || user()?.role !== "admin" || !datasetId()) return;
        openedLink = true;
        try { await open(requested); }
        catch (error) { if (!error.stale) workspace.notice(error.message, true); }
    });
    if (canvas) window.addEventListener("storage", event => { if (event.key === "tator.hosted.session-event" && nav) nav.hidden = true; });
    mount();
})();
