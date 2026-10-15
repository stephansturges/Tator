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
            if (!conflicts.length) { notify("No conflicting imported labels for this image."); negativeLabel.hidden = true; return; }
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
        window.addEventListener("tator:hosted-image", () => { if (imageId !== app()?.imageId) { ++ticket; dialog.close(); } });
    }, {once: true});
})();
