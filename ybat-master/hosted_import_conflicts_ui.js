/* Human review of imported labels; the canvas owns saving and reply recovery. */
(() => {
    "use strict";
    if (window.TatorHostedBoot?.user.role !== "admin") return;
    const node = (tag, text) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; return element; };
    let button, dialog, imageId = "", ticket = 0, currentRecord, conflicts = [], decisions = [], classNames = new Map();
    const heading = node("h2", "Compare imported labels"), notice = node("p"), choice = node("select"), comparison = node("div");
    const chooser = node("label", "Imported label version"); chooser.append(choice);
    const negative = node("input"); negative.type = "checkbox"; negative.id = "importConflictNegative";
    const negativeLabel = node("label"); negativeLabel.append(negative, document.createTextNode(" I checked the image: it contains no target objects."));
    const keep = node("button", "Keep current labels"), use = node("button", "Use imported objects"), close = node("button", "Close");
    notice.setAttribute("role", "status");
    const app = () => window.TatorHosted;
    function notify(text, error = false) { notice.textContent = text; notice.dataset.error = String(error); }
    function eligible() { return dialog?.open && app()?.imageId === imageId; }
    function selected() { return conflicts.find(item => item.id === choice.value); }
    function controls() {
        const conflict = selected(), resolved = decisions.some(item => item.conflict_id === conflict?.id);
        keep.disabled = !conflict || resolved || !app()?.editable;
        use.disabled = keep.disabled || (!conflict.annotations.objects.length && !negative.checked);
    }
    async function draw() {
        const conflict = selected(); if (!conflict || !eligible()) return;
        const seq = ++ticket; comparison.replaceChildren(); negative.checked = false;
        negativeLabel.hidden = Boolean(conflict.annotations.objects.length);
        const reviewed = decisions.find(item => item.conflict_id === conflict.id);
        notify(reviewed ? `Reviewed: ${reviewed.choice === "keep_current" ? "current labels kept" : "imported objects used"}. Both versions remain in history.` : app().editable ? "Compare the boxes, then choose which objects to keep. Your current captions and other metadata are preserved." : "Comparison is read-only. Reconnect editing, or recall this image's assignment before changing its labels.");
        controls();
        const photo = new Image();
        photo.src = `/api/hosted/images/${encodeURIComponent(imageId)}/media`;
        await photo.decode();
        if (seq !== ticket || !eligible()) return;
        for (const [title, annotations, color] of [["Current saved labels", currentRecord.annotations, "#21b6e8"], ["Imported labels", conflict.annotations, "#ff9d43"]]) {
            const panel = node("section"), canvas = node("canvas"), objects = annotations.objects || [];
            const scale = Math.min(600 / photo.naturalWidth, 420 / photo.naturalHeight);
            canvas.width = Math.max(1, Math.round(photo.naturalWidth * scale)); canvas.height = Math.max(1, Math.round(photo.naturalHeight * scale));
            canvas.style.width = "100%"; canvas.style.height = "auto";
            const context = canvas.getContext("2d"); context.drawImage(photo, 0, 0, canvas.width, canvas.height);
            context.strokeStyle = color; context.fillStyle = color; context.lineWidth = 2; context.font = "12px sans-serif";
            const counts = new Map();
            for (const object of objects) {
                const [x, y, width, height] = object.bbox, name = classNames.get(object.class_id) || "Unknown class";
                counts.set(name, (counts.get(name) || 0) + 1);
                context.strokeRect(x * scale, y * scale, width * scale, height * scale);
                if (object.polygon?.length) { context.beginPath(); object.polygon.forEach(([px, py], index) => index ? context.lineTo(px * scale, py * scale) : context.moveTo(px * scale, py * scale)); context.closePath(); context.stroke(); }
                context.fillText(name, x * scale + 2, Math.max(12, y * scale - 3));
            }
            panel.append(node("h3", `${title} · ${objects.length} object${objects.length === 1 ? "" : "s"}`), canvas,
                node("p", [...counts].map(([name, amount]) => `${amount} ${name}`).join(" · ") || "No annotated objects"));
            comparison.append(panel);
        }
    }
    async function open() {
        if (!app() || !button) return;
        imageId = app().imageId; const seq = ++ticket; if (!dialog.open) dialog.showModal(); comparison.replaceChildren(); notify("Loading saved and imported labels…");
        keep.disabled = use.disabled = true;
        try {
            const [record, history] = await Promise.all([app().api(`/api/hosted/images/${encodeURIComponent(imageId)}`), app().api(`/api/hosted/images/${encodeURIComponent(imageId)}/history`)]);
            if (seq !== ticket || !eligible()) return;
            const dataset = await app().api(`/api/hosted/datasets/${encodeURIComponent(record.dataset_id)}`);
            if (seq !== ticket || !eligible()) return;
            currentRecord = record; conflicts = history.import_conflicts || []; decisions = history.import_conflict_decisions || [];
            classNames = new Map(dataset.classes.map(item => [item.id, item.name])); heading.textContent = `Imported labels · ${record.filename}`;
            choice.replaceChildren();
            for (const item of conflicts) { const option = node("option", `${item.filename} · ${new Date(item.created_at * 1000).toLocaleString()}${decisions.some(decision => decision.conflict_id === item.id) ? " · Reviewed" : " · Needs review"}`); option.value = item.id; choice.append(option); }
            const unresolved = conflicts.find(item => !decisions.some(decision => decision.conflict_id === item.id));
            if (unresolved) choice.value = unresolved.id;
            if (!conflicts.length) {
                // An import can mark an image for review without keeping a second label set
                // (imports before importer v5). The decision is then made on the canvas itself.
                const marked = record.annotations?.migration?.review_required;
                const reason = {duplicate_labels_differ: "its duplicate copies had different labels"}[marked] || (typeof marked === "string" ? marked.replaceAll("_", " ") : "");
                notify(marked ? `The import marked this image for a decision${reason ? `: ${reason}` : ""}. No other label set was kept to compare, so check the boxes on the canvas, correct them if needed, then mark Annotations checked and approve it. Approving ends the decision.` : "No conflicting imported labels for this image.");
                negativeLabel.hidden = true; return;
            }
            await draw();
        } catch (error) { if (eligible()) notify(error.message, true); }
    }
    async function apply(decision) {
        const conflict = selected(); if (!eligible() || !conflict) return;
        keep.disabled = use.disabled = choice.disabled = true;
        try {
            await app().applyImportConflict(conflict.id, {choice: decision, negative: negative.checked, expectedVersion: currentRecord.current_version});
            if (eligible()) await open();
        } catch (error) { if (eligible()) notify(`${error.message} Your saved versions remain available.`, true); }
        finally { choice.disabled = false; controls(); }
    }
    // "Needs a decision" from Incoming is explained on the canvas: why, and
    // what to do. Approving the image is the decision for an import review.
    const decisionReasons = {duplicate_labels_differ: "duplicate copies of this image had different labels; the copy with the most objects was kept"};
    let decisionTicket = 0;
    async function refreshDecision() {
        const id = app()?.imageId; if (!id || !button) return;
        const seq = ++decisionTicket;
        try {
            const [record, history] = await Promise.all([app().api(`/api/hosted/images/${encodeURIComponent(id)}`), app().api(`/api/hosted/images/${encodeURIComponent(id)}/history`)]);
            if (seq !== decisionTicket || app()?.imageId !== id) return;
            const reviewed = new Set((history.import_conflict_decisions || []).map(item => item.conflict_id));
            const unresolved = (history.import_conflicts || []).filter(item => !reviewed.has(item.id)).length;
            const flag = record.annotations?.migration?.review_required;
            const reason = typeof flag === "string" && flag ? decisionReasons[flag] || flag.replaceAll("_", " ") : flag === true ? "the import marked it for review" : "";
            button.textContent = unresolved ? `Imported labels · ${unresolved} to review` : "Imported labels";
            document.getElementById("hostedDecisionNote")?.remove();
            if (!unresolved && !reason) return;
            const note = node("div"); note.id = "hostedDecisionNote"; note.className = "hosted-decision-note"; note.setAttribute("role", "note");
            const why = [reason, unresolved ? `${unresolved === 1 ? "an imported label version differs" : `${unresolved} imported label versions differ`} from the saved labels` : ""].filter(Boolean).join("; ");
            const finish = record.state === "unassigned" ? " Then tick Annotations checked and approve the image." : "";
            note.append(node("strong", "Needs a decision: "), document.createTextNode(`${why}.${unresolved ? " Compare them and keep one version." : " Check the boxes."}${finish}`));
            if (unresolved) { const compare = node("button", "Compare imported labels"); compare.type = "button"; compare.onclick = () => open(); note.append(document.createElement("br"), compare); }
            const anchor = document.getElementById("hostedReviewNotes");
            if (anchor) anchor.after(note);
        } catch (_) { /* The Imported labels button still works. */ }
    }
    window.addEventListener("load", () => {
        button = node("button", "Imported labels"); button.id = "hostedImportConflicts"; button.type = "button";
        const attach = () => { const target = document.querySelector(".hosted-canvas-feedback"); if (!target) return false; target.append(button); return true; };
        if (!attach()) { const observer = new MutationObserver(() => { if (attach()) observer.disconnect(); }); observer.observe(document.body, {childList: true, subtree: true}); }
        dialog = node("dialog"); dialog.id = "importConflictsDialog"; dialog.style.width = "min(95vw, 1000px)";
        comparison.style.display = "grid"; comparison.style.gridTemplateColumns = "repeat(auto-fit,minmax(min(100%,300px),1fr))"; comparison.style.gap = "1rem";
        const actions = node("div"); actions.className = "dialog-actions"; actions.append(close, keep, use);
        for (const control of [close, keep, use]) control.type = "button";
        dialog.append(heading, chooser, notice, comparison, negativeLabel, actions); document.body.append(dialog);
        button.onclick = open; choice.onchange = () => draw().catch(error => notify(error.message, true)); negative.onchange = controls;
        close.onclick = () => dialog.close(); keep.onclick = () => apply("keep_current"); use.onclick = () => apply("use_imported_objects");
        dialog.addEventListener("close", () => { ++ticket; });
        // The canvas page claims Escape for drawing; while this dialog is open, Escape closes it.
        document.addEventListener("keydown", event => { if (event.key === "Escape" && dialog.open) { event.preventDefault(); event.stopPropagation(); dialog.close(); } }, true);
        window.addEventListener("tator:hosted-image", () => { if (imageId !== app()?.imageId) { ++ticket; dialog.close(); } refreshDecision(); });
        window.addEventListener("tator:import-conflict-resolved", refreshDecision);
        // The first image: wait until the canvas has its record.
        const first = setInterval(() => { if (app()?.datasetId && document.getElementById("hostedReviewNotes")) { clearInterval(first); refreshDecision(); } }, 300);
    }, {once: true});
})();
