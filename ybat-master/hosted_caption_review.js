/* Explicit administrator review of a retained, version-bound caption proposal. */
(() => {
    "use strict";
    const batchId = new URLSearchParams(location.search).get("caption_batch");
    if (!batchId || window.TatorHostedBoot?.user.role !== "admin") return;
    const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
    window.addEventListener("load", async () => {
        let panel;
        try {
            const began = Date.now();
            while (!window.TatorHostedCanvas?.ready()) {
                if (Date.now() - began > 30000) return;
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            const app = window.TatorHosted, imageId = app.imageId;
            const proposal = await app.api(`/api/hosted/tools/caption-batches/${encodeURIComponent(batchId)}/images/${encodeURIComponent(imageId)}/proposal`);
            if (app.imageId !== imageId) return;
            const record = await app.api(`/api/hosted/images/${encodeURIComponent(imageId)}`);
            if (app.imageId !== imageId) return;
            panel = node("section"); panel.className = "hosted-caption-review"; panel.id = "hostedCaptionProposal";
            panel.append(node("h2", "Proposed caption"), node("p", proposal.caption || "This attempt did not produce a caption."));
            const note = node("p"); note.setAttribute("role", "status"); panel.append(note);
            const includeQa = node("input"); includeQa.type = "checkbox"; includeQa.checked = true;
            const qaLabel = node("label"); qaLabel.append(includeQa, document.createTextNode(` Include ${proposal.generated_qa_pairs?.length || 0} generated question-and-answer pairs`));
            if (proposal.generated_qa_pairs?.length) panel.append(qaLabel);
            if (proposal.generated_qa_pairs?.length) {
                const pairs = node("details"); pairs.append(node("summary", "Read the proposed questions and answers"));
                for (const pair of proposal.generated_qa_pairs) {
                    pairs.append(node("strong", pair.question || "Question"), node("p", pair.answer || ""));
                }
                panel.append(pairs);
            }
            const primary = node("input"); primary.type = "checkbox"; primary.checked = true;
            const label = node("label"); label.append(primary, document.createTextNode(" Use as primary caption; preserve the previous caption as an alternate")); panel.append(label);
            if (record.current_version !== proposal.source_version) note.textContent = "This image has changed since the run. The proposal remains available for comparison; generate a new caption before applying it.";
            const apply = node("button", "Use this caption"); apply.id = "hostedUseCaption";
            apply.disabled = record.current_version !== proposal.source_version || !proposal.caption || !app.editable;
            apply.onclick = async () => {
                apply.disabled = true;
                try { await app.applyCaptionProposal(batchId, proposal, {includeGeneratedQa: includeQa.checked, replacePrimary: primary.checked}); }
                catch (error) { note.textContent = error.message; apply.disabled = !app.editable; }
            };
            panel.append(apply); document.querySelector(".hosted-canvas-feedback").after(panel);
            window.addEventListener("tator:hosted-image", () => { if (app.imageId !== imageId) panel.remove(); });
            window.addEventListener("tator:caption-applied", () => { note.textContent = "Saved. Your earlier captions remain available as alternates."; apply.disabled = true; }, {once: true});
        } catch (error) {
            if (!panel) { panel = node("p"); panel.className = "hosted-caption-review"; document.querySelector(".hosted-canvas-feedback")?.after(panel); }
            panel.append(node("span", `Caption proposal unavailable: ${error.message}`));
        }
    }, {once: true});
})();
