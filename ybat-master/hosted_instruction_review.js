/* Human QA decisions remain versioned annotations, never automatic readiness. */
(() => {
    "use strict";
    if (window.TatorHostedBoot?.user.role !== "admin") return;
    const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
    const labels = {unreviewed: "Not reviewed", accepted: "Accept", needs_revision: "Needs correction", rejected: "Reject"};
    let dialog, notice, content, save, review, controls = [], epoch = 0;

    function build() {
        if (dialog) return;
        dialog = node("dialog"); dialog.id = "hostedInstructionReviewDialog";
        dialog.style.cssText = "width:min(940px,94vw);max-height:90vh;overflow:auto;padding:24px";
        notice = node("p"); notice.setAttribute("role", "status");
        content = node("div"); save = node("button", "Save review decisions"); save.id = "hostedSaveInstructionReview";
        const close = node("button", "Close"); close.type = "button"; close.onclick = () => dialog.close();
        dialog.append(node("h2", "Review captions and QA"), node("p", "Check each answer against the image. Accept, reject, or mark it for correction. These decisions are saved with your identity; model evidence stays intact. Export still checks language quality and dataset coverage."), notice, content, save, close);
        dialog.addEventListener("close", () => { ++epoch; });
        save.onclick = async () => {
            const ticket = epoch, app = window.TatorHosted;
            const decisions = controls.filter(row => row.choice.value !== row.originalDecision || row.notes.value !== row.originalNotes)
                .map(row => ({target_id: row.target, decision: row.choice.value, notes: row.notes.value}));
            if (!decisions.length) { notice.textContent = "No decisions changed."; return; }
            save.disabled = true;
            try {
                await app.applyInstructionReview(review, decisions);
                if (ticket !== epoch || !dialog.open) return;
                notice.textContent = "Review saved. Create a new snapshot when you export; existing snapshots keep their original review state.";
                await load(ticket, false);
            } catch (error) { if (ticket === epoch) notice.textContent = error.message; }
            finally { if (ticket === epoch) save.disabled = !app.editable; }
        };
        document.body.append(dialog);
    }

    async function load(ticket, flush = true) {
        const app = window.TatorHosted, imageId = app.imageId;
        if (flush && app.editable && !await app.save({manual: true})) throw new Error("Save this image before reviewing its captions.");
        if (ticket !== epoch || imageId !== app.imageId) return;
        const value = await app.api(`/api/hosted/tools/images/${encodeURIComponent(imageId)}/instruction-review`);
        if (ticket !== epoch || imageId !== app.imageId || !dialog.open) return;
        review = value; controls = []; content.replaceChildren();
        const image = node("img"); image.src = app.mediaUrl; image.alt = "Image being reviewed";
        image.style.cssText = "display:block;width:100%;max-height:260px;object-fit:contain;margin-bottom:16px"; content.append(image);
        if (!value.rows.length) content.append(node("p", "This image has no saved captions or question-and-answer pairs."));
        for (const row of value.rows) {
            const item = node("fieldset"); item.style.cssText = "padding:16px;margin:12px 0"; item.dataset.target = row.target_id;
            item.append(node("legend", row.kind === "caption0" ? (row.is_primary ? "Primary caption" : "Alternate caption") : "Question and answer"));
            if (row.kind === "generated_qa") item.append(node("strong", row.question));
            const answer = node("p", row.answer); answer.style.whiteSpace = "pre-wrap"; item.append(answer);
            if (row.text_changed_since_review) item.append(node("p", "This text changed after its last review. Review the current wording again."));
            if (row.text_changed_since_machine_validation) item.append(node("p", "The model checked earlier wording. Its recorded validation does not qualify this edited answer."));
            if (row.validation_status || row.rejection_reasons?.length) item.append(node("small", `Recorded model validation: ${row.validation_status || "unspecified"}${row.rejection_reasons?.length ? " · " + row.rejection_reasons.join(", ") : ""}`));
            const choice = node("select"); choice.setAttribute("aria-label", `Review decision for ${row.target_id}`);
            for (const [value, label] of Object.entries(labels)) choice.add(new Option(label, value));
            choice.value = labels[row.decision] ? row.decision : "unreviewed";
            const label = node("label", "Your decision "); label.append(choice); item.append(label);
            const notes = node("textarea"); notes.rows = 2; notes.maxLength = 4000; notes.value = row.notes || "";
            notes.setAttribute("aria-label", `Review note for ${row.target_id}`); notes.placeholder = "Reason or correction needed (optional)";
            notes.style.cssText = "display:block;width:100%;margin-top:8px"; item.append(notes);
            if (row.reviewed_at) item.append(node("small", `Last reviewed ${new Date(row.reviewed_at).toLocaleString()} · ${row.reviewer}`));
            choice.disabled = notes.disabled = !app.editable;
            controls.push({target: row.target_id, choice, notes, originalDecision: choice.value, originalNotes: notes.value}); content.append(item);
        }
        save.disabled = !app.editable || !value.rows.length;
        if (!app.editable) notice.textContent = "Read-only view. Reconnect administrator editing to save review decisions.";
    }

    function installButton() {
        if (document.getElementById("hostedReviewInstructions")) return;
        const button = node("button", "Review captions and QA"); button.id = "hostedReviewInstructions"; button.type = "button";
        button.onclick = async () => {
            build(); const ticket = ++epoch; notice.textContent = "Loading saved language annotations…";
            content.replaceChildren(); save.disabled = true; dialog.showModal();
            try { await load(ticket); if (ticket === epoch && window.TatorHosted.editable) notice.textContent = "Choose a decision for each answer you have reviewed."; }
            catch (error) { if (ticket === epoch) notice.textContent = error.message; }
        };
        document.querySelector(".hosted-canvas-feedback")?.append(button);
    }
    window.addEventListener("load", installButton, {once: true});
    window.addEventListener("tator:hosted-canvas", installButton);
    window.addEventListener("tator:hosted-image", () => { if (dialog?.open) dialog.close(); });
})();
