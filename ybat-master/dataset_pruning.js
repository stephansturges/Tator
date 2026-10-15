(function (root) {
    "use strict";
    const h = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
    const btn = (text, action) => { const n = h("button", "", text); n.type = "button"; n.addEventListener("click", action); return n; };
    const pct = n => `${(Number(n) * 100).toFixed(1)}%`;
    const LABELS = [["", "Choose a judgment for the left image"], ["removable_under_policy", "Could remove left · same useful views"], ["useful_variation_keep", "Keep left · useful difference"], ["wrong_correspondence_keep", "Keep left · wrong object comparison"], ["uncertain", "Not sure"]];
    const name = v => v.id === "historical" ? "Raw C-RADIO · same compact branch" : v.id === "candidate" ? "Full C-RADIO correction · same compact branch" : "Your comparison settings";
    function create(io) {
        const box = io.host.querySelector('[data-pruning="recipeLab"]');
        if (!box) return null;
        let meta = null, comparison = null, sequence = 0, busy = false, revealed = false;
        let draft = { alpha: .5, cradio_weight: .4 }, judgments = new Map(), filter = "all";
        const title = h("h3", "", "Compare recipes and choose their cutoffs");
        const intro = h("p", "pruning-help", "The historical Thorough recipe corrects DINOv3, SAM3 and crop SALAD, but leaves C-RADIO raw. Compare that choice with partial or full C-RADIO correction using the same cached features. Review the left image before revealing scores; these judgments choose cutoffs and do not approve removals. The WALDO40 report suggests investigating half correction with a 40% C-RADIO object share; this is a comparison starting point, with cutoffs still to review. Add another example from a family’s exact-bbox comparison, or remove an example from this list.");
        const load = btn("Load comparison examples", () => refresh(io.state.savedLab));
        const status = h("p", "pruning-status"); status.setAttribute("role", "status");
        const setup = h("div", "pruning-rule-grid");
        const alpha = h("input"), weight = h("input");
        for (const [n, value, max] of [[alpha, .5, 1], [weight, .4, .95]]) { n.type = "number"; n.min = "0"; n.max = String(max); n.step = ".05"; n.value = value; }
        const field = (label, control, help) => { const n = h("label", "", label); n.append(control, h("span", "pruning-help", help)); return n; };
        setup.append(field("C-RADIO correction strength", alpha, "0 leaves its fingerprint raw. 0.5 subtracts half of the fitted size-and-shape prediction. 1 subtracts all of it. Tator then normalizes the complete vector."), field("C-RADIO share of object similarity", weight, "0.25 gives C-RADIO 25%. The remaining 75% keeps the existing DINOv3 / SAM3 / crop SALAD proportions. Whole-image settings stay fixed."));
        const actions = h("div", "pruning-inline");
        const reveal = btn("Reveal recipe comparisons", compare);
        const fit = btn("Fit cutoffs for these settings", calibrate);
        const selectFilter = h("select"); selectFilter.add(new Option("All reviewed examples", "all")); selectFilter.add(new Option("Only recipe disagreements", "disagreements"));
        actions.append(reveal, fit, field("Show", selectFilter, "Disagreements depend on each recipe's cutoffs. An uncalibrated recipe uses the active analysis cutoffs only for orientation."));
        const outcome = h("div", "pruning-lab-outcome"), profiles = h("div", "pruning-lab-profiles"), list = h("div", "pruning-lab-list");
        box.replaceChildren(title, intro, load, status, setup, actions, outcome, profiles, list);
        function anchor() { return JSON.stringify([io.state.context, io.state.jobId, io.state.result?.selection_revision, meta?.base_digest, draft]); }
        function current(ticket, scope) { return sequence === ticket && anchor() === scope && io.contextIsCurrent(io.state.context); }
        function blocked() { return busy || io.state.busy || io.state.exporting || io.state.externalRefreshing || io.state.recoveryRequired || !!io.state.pending; }
        function sync() {
            const lock = blocked();
            load.disabled = lock || !io.state.result;
            setup.hidden = actions.hidden = !meta?.available;
            alpha.disabled = weight.disabled = lock;
            reveal.disabled = lock || !meta?.available || !meta.pairs.length;
            fit.disabled = lock || !meta?.available || !meta.pairs.length || meta.pairs.some(p => !judgments.get(p.case_id)?.label);
            selectFilter.disabled = lock || !comparison;
            box.querySelectorAll('[data-lab-edit],[data-lab-apply]').forEach(n => { n.disabled = lock; });
        }
        function snapshot() {
            return meta ? { jobId: meta.job_id, baseDigest: meta.base_digest, draft, revealed, labels: Array.from(judgments), pairs: meta.pairs, filter } : null;
        }
        function saved() { io.remember(); sync(); }
        function pairs() { return meta.pairs.map(({ case_id, candidate_id, witness_id }) => ({ case_id, candidate_id, witness_id })); }
        function value() {
            const a = Number(alpha.value), w = Number(weight.value);
            if (!Number.isFinite(a) || a < 0 || a > 1 || !Number.isFinite(w) || w < 0 || w >= 1) throw new Error("Choose correction strength from 0 to 1 and a C-RADIO share below 1.");
            return { id: "custom", alpha: a, cradio_weight: w };
        }
        function payload() { return { expected_digest: meta.base_digest, selection_revision: meta.selection_revision, pairs: pairs() }; }
        function renderProfiles() {
            profiles.replaceChildren();
            for (const profile of meta?.profiles || []) {
                const card = h("div", "pruning-lab-profile");
                const gates = profile.thresholds;
                if (profile.review_protocol) card.append(h("p", "pruning-help", profile.review_protocol.reported_scores_visible_before_all_judgments === true ? "Scores were visible before at least one submitted judgment. This review is not score-blinded." : profile.review_protocol.reported_scores_visible_before_all_judgments === false ? "The browser records judgments made before scores were revealed; this is a submitted review, not an independent blindness audit." : "Score exposure was not recorded; this is submitted review, with no independent blindness audit."));
                card.append(h("strong", "", gates ? `Saved cutoffs · object ${pct(gates.object_threshold)}, scene ${pct(gates.scene_threshold)}, index ${pct(gates.index_threshold)}` : "Review saved; more evidence needed"), h("p", "pruning-help", profile.reason || profile.qualification));
                if (profile.variant) card.append(h("p", "pruning-help", `C-RADIO correction ${pct(profile.variant.alpha)} · share ${pct(profile.variant.cradio_weight)}`));
                if (profile.available && profile.draft_config) {
                    const apply = btn("Use this recipe and its cutoffs in setup", () => {
                        if (blocked() || !io.contextIsCurrent(io.state.context)) return;
                        io.applyConfig({ ...profile.draft_config, calibration_profile: profile.reference, threshold_qualification: "diagnostic_calibrated" });
                        io.notify("Recipe and its saved cutoffs copied to setup. Analyze the draft to recompute families; removal approvals are separate.");
                        saved();
                    }); apply.dataset.labApply = ""; card.append(apply);
                } else if (gates) card.append(h("p", "pruning-help", "This profile is unavailable for the current source or saved feature recipe."));
                profiles.append(card);
            }
        }
        function renderOutcome() {
            outcome.replaceChildren();
            if (!comparison) return;
            outcome.append(h("p", "", `${comparison.disagreements.length} of ${meta.pairs.length} directions receive different decisions across the recipes.`));
            for (const variant of comparison.variants) {
                const t = variant.thresholds, count = variant.rows.filter(r => r.flagged).length;
                outcome.append(h("p", "pruning-help", `${name(variant)} · C correction ${pct(variant.alpha)} / share ${pct(variant.cradio_weight)} · ${count} flagged directions · floors: object ${pct(t.object_threshold)}, scene ${pct(t.scene_threshold)}, index ${pct(t.index_threshold)}. ${variant.threshold_qualification === "diagnostic_calibrated" ? "Uses its own saved diagnostic cutoffs." : "Uses the active job's cutoffs; this variant has not been calibrated."}`));
            }
            outcome.append(h("p", "pruning-help", "A flagged direction says the left image could be covered by the right under these rules. It is not a final family removal count. Selected examples cannot establish population accuracy."));
        }
        function cropImage(url, label) { const img = h("img"); img.src = io.assetURL(url, io.state.context); img.alt = label; img.loading = "lazy"; return img; }
        function renderList() {
            list.replaceChildren();
            const disagreements = new Set(comparison?.disagreements.map(v => v.case_id) || []);
            for (const pair of meta?.pairs || []) {
                if (filter === "disagreements" && !disagreements.has(pair.case_id)) continue;
                const card = h("article", "pruning-lab-card"); card.dataset.caseId = pair.case_id;
                card.append(h("h4", "", `Could the right image replace the useful labelled views on the left?`));
                const remove = btn("Remove this example from cutoff review", () => { if (blocked()) return; meta.pairs = meta.pairs.filter(p => p.case_id !== pair.case_id); judgments.delete(pair.case_id); comparison = null; filter = "all"; selectFilter.value = filter; renderOutcome(); renderList(); saved(); }); remove.dataset.labEdit = ""; card.append(remove);
                const images = h("div", "pruning-comparison-images");
                images.append(io.thumbnail(pair.candidate, io.state.context, "Left · candidate to judge", true), io.thumbnail(pair.witness, io.state.context, "Right · image that would remain", true)); card.append(images);
                const crops = h("details", "pruning-disclosure"); crops.append(h("summary", "", "Inspect every labelled object before judging"));
                for (const side of ["candidate", "witness"]) {
                    const row = h("div", "pruning-lab-native-crops"); row.append(h("strong", "", side === "candidate" ? "Left image" : "Right image"));
                    for (const object of pair[`${side}_objects`] || []) row.append(cropImage(object.url, `${side} · ${io.className(object.class_id)} · row ${object.row}`));
                    crops.append(row);
                } card.append(crops);
                const judgment = h("select"); for (const [id, text] of LABELS) judgment.add(new Option(text, id)); judgment.value = judgments.get(pair.case_id)?.label || ""; judgment.dataset.labEdit = "";
                const note = h("input"); note.type = "text"; note.maxLength = 2000; note.placeholder = "What changed, or why is this redundant?"; note.value = judgments.get(pair.case_id)?.reason || ""; note.dataset.labEdit = "";
                const update = () => { judgments.set(pair.case_id, { case_id: pair.case_id, label: judgment.value, reason: note.value, scores_visible_when_judged: revealed }); saved(); };
                judgment.addEventListener("change", update); note.addEventListener("input", update);
                card.append(field("Your judgment concerns the left image", judgment, "This review fits similarity cutoffs; it does not approve a dataset removal."), field("Reason", note, "Useful changes can include angle, pose, glare, occlusion or clearer detail."));
                if (revealed && comparison) for (const variant of comparison.variants) {
                    const measured = variant.rows.find(v => v.case_id === pair.case_id);
                    if (!measured) continue;
                    const detail = h("details", "pruning-disclosure");
                    detail.append(h("summary", "", `${name(variant)}: ${measured.flagged ? "flagged" : "kept"} · index ${pct(measured.score)} · scene ${pct(measured.scene)}`));
                    const matches = h("div", "pruning-crop-grid");
                    for (const crop of measured.crops || []) {
                        const row = h("div", "pruning-crop-row"), imgs = h("div", "pruning-crop-pair");
                        imgs.append(cropImage(crop.candidate_url, "Left matched bbox"), cropImage(crop.witness_url, "Right matched bbox"));
                        row.append(h("strong", "", io.className(crop.class_id)), imgs, h("p", "pruning-help", `Rows ${crop.candidate_row} → ${crop.witness_row} · appearance ${pct(crop.appearance)} · scale ${pct(crop.scale)}`)); matches.append(row);
                    }
                    detail.append(matches); card.append(detail);
                }
                list.append(card);
            }
            if (meta?.pairs.length && !list.children.length) list.append(h("p", "pruning-help", "No disagreements among these examples. Try another correction strength or weight, or calibrate each variant."));
            sync();
        }
        async function refresh(restore = null) {
            if (!io.state.result || blocked()) return;
            const ticket = ++sequence, context = io.state.context, job = io.state.jobId, revision = io.state.result.selection_revision;
            busy = true; sync(); status.textContent = "Loading cached comparison examples …";
            try {
                const data = await io.request(`/jobs/${encodeURIComponent(job)}/recipe-lab`, {}, context);
                if (ticket !== sequence || context !== io.state.context || job !== io.state.jobId || revision !== io.state.result?.selection_revision) return;
                const previous = restore || snapshot();
                meta = data; comparison = null; judgments = new Map(); filter = "all";
                if (previous?.jobId === job && previous.baseDigest === data.base_digest) {
                    if (Array.isArray(previous.pairs) && previous.pairs.length <= 24) meta.pairs = previous.pairs;
                    const ids = new Set(meta.pairs.map(v => v.case_id)); judgments = new Map((previous.labels || []).filter(([id]) => ids.has(id)));
                    draft = previous.draft || draft; revealed = !!previous.revealed;
                } else revealed = false;
                alpha.value = draft.alpha; weight.value = draft.cradio_weight; selectFilter.value = filter;
                status.textContent = data.available ? `${data.pairs.length} selected directions. Review every left image, including its labelled objects. ${data.scope}` : data.reason;
                renderProfiles(); renderOutcome(); renderList(); saved();
            } catch (error) { if (ticket === sequence && context === io.state.context && error.name !== "AbortError") status.textContent = error.message; }
            finally { if (ticket === sequence) { busy = false; sync(); } }
        }
        async function compare() {
            if (!meta?.available || blocked()) return;
            const ticket = ++sequence, scope = anchor(), context = io.state.context, job = io.state.jobId;
            try {
                const variant = value(), variants = [meta.baseline, meta.candidate, variant];
                busy = true; sync(); status.textContent = "Reusing cached object features and recomputing same-class assignments …";
                const data = await io.request(`/jobs/${encodeURIComponent(job)}/recipe-lab/compare`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...payload(), variants }) }, context);
                if (!current(ticket, scope)) return;
                comparison = data; revealed = true; status.textContent = data.qualification;
                renderOutcome(); renderList(); saved();
            } catch (error) { if (current(ticket, scope) && error.name !== "AbortError") status.textContent = error.message; }
            finally { if (ticket === sequence) { busy = false; sync(); } }
        }
        async function calibrate() {
            if (!meta?.available || blocked()) return;
            const ticket = ++sequence, scope = anchor(), context = io.state.context, job = io.state.jobId;
            try {
                const variant = value(), labels = meta.pairs.map(p => judgments.get(p.case_id));
                if (labels.some(v => !v?.label)) throw new Error("Review every left image first. Choose Not sure when you cannot decide.");
                busy = true; io.state.busy = true; io.controls(); sync();
                status.textContent = "Fitting this recipe's cutoffs from your judgments, with fresh object assignments at each searched floor …";
                const data = await io.mutation("calibration", `/jobs/${encodeURIComponent(job)}/recipe-lab/calibrate`, { ...payload(), variant, labels, review_protocol: { reported_scores_visible_before_all_judgments: labels.some(v => v.scores_visible_when_judged === true), score_visibility_recorded: true } }, context);
                if (!current(ticket, scope)) return;
                const profile = data.profile;
                meta.profiles = [...meta.profiles.filter(p => p.profile_id !== profile.profile_id), { ...profile, available: profile.status === "diagnostic_calibrated", variant: data.variant }];
                comparison = null; renderProfiles(); renderOutcome(); renderList();
                status.textContent = profile.status === "diagnostic_calibrated" ? "Saved diagnostic cutoffs for this exact recipe. Use the profile button to copy the complete recipe into setup, or fit another variant before comparing." : profile.reason;
                saved();
            } catch (error) { if (current(ticket, scope) && error.name !== "AbortError") status.textContent = error.message; }
            finally { if (ticket === sequence) { busy = false; io.state.busy = false; io.controls(); sync(); io.onSettled?.(); } }
        }
        for (const n of [alpha, weight]) n.addEventListener("input", () => { draft = { alpha: Number(alpha.value), cradio_weight: Number(weight.value) }; comparison = null; filter = "all"; selectFilter.value = filter; renderOutcome(); renderList(); saved(); });
        selectFilter.addEventListener("change", () => { filter = selectFilter.value; renderList(); saved(); });
        async function addPair(candidateId, witnessId) {
            if (blocked()) return;
            if (!meta) await refresh(io.state.savedLab);
            if (!meta?.available || blocked()) { io.notify(meta?.reason || "Load a complete Thorough object recipe before adding examples.", true); return; }
            const ticket = ++sequence, scope = anchor(), context = io.state.context, job = io.state.jobId;
            busy = true; sync();
            try {
                const data = await io.request(`/jobs/${encodeURIComponent(job)}/recipe-lab/pair`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_digest: meta.base_digest, selection_revision: meta.selection_revision, candidate_id: candidateId, witness_id: witnessId }) }, context);
                if (!current(ticket, scope)) return;
                if (meta.pairs.some(p => p.case_id === data.pair.case_id)) { io.notify("This direction is already in the cutoff review."); return; }
                if (meta.pairs.length >= 24) throw new Error("Remove one example from the cutoff review before adding another; up to 24 directions are supported.");
                meta.pairs.push(data.pair); comparison = null; filter = "all"; selectFilter.value = filter;
                renderOutcome(); renderList(); saved(); io.notify("Image pair added to the recipe cutoff review. Judge its left image there; removal approvals are separate.");
            } catch (error) { if (current(ticket, scope) && error.name !== "AbortError") io.notify(error.message, true); }
            finally { if (ticket === sequence) { busy = false; sync(); } }
        }
        function reset() { sequence++; busy = false; meta = comparison = null; judgments.clear(); list.replaceChildren(); profiles.replaceChildren(); outcome.replaceChildren(); status.textContent = "Complete a compound analysis with all four Thorough object components to compare cached recipes."; sync(); }
        reset();
        return { sync, reset, snapshot, addPair, restore: async savedState => { await refresh(savedState); }, refresh };
    }
    root.TatorPruningRecipeLab = Object.freeze({ create });
}(typeof globalThis !== "undefined" ? globalThis : this));

(function (root) {
    "use strict";
    const NS = "http://www.w3.org/2000/svg";
    const pct = (v, digits = 1) => v !== null && v !== undefined && Number.isFinite(Number(v)) ? `${(100 * Number(v)).toFixed(digits)}%` : "unmeasured";
    const number = v => v === undefined || v === null ? "unmeasured" : Number(v).toLocaleString();
    const digest = v => JSON.stringify(v);
    const nonce = () => crypto.randomUUID().replaceAll("-", "");
    const h = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
    const button = (text, action, cls = "") => { const n = h("button", cls, text); n.type = "button"; if (action) n.addEventListener("click", action); return n; };
    const input = (type, value, attrs = {}) => { const n = h("input"); n.type = type; n.value = value; Object.assign(n, attrs); return n; };
    const field = (text, control, help) => { const n = h("label", "", text); n.append(control); if (help) n.append(h("span", "pruning-help", help)); return n; };
    const select = (values, value) => { const n = h("select"); for (const [id, label] of values) n.add(new Option(label, id)); if (value !== undefined) n.value = value; return n; };
    const idOf = image => image.image_id || image.id;
    const sourceKey = c => c.annotationSource?.id ? `${c.annotationSource.kind}:${c.annotationSource.id}` : "";
    const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
    const cleanContext = c => ({ ...c, apiRoot: String(c?.apiRoot || "").replace(/\/$/, "") });
    const signature = c => digest([c?.apiRoot, c?.backendId, c?.analysisJobId || "", c?.annotationSource, c?.revision]);

    function create(adapters) {
        const host = document.getElementById("datasetPruning");
        if (!host) return null;
        if (host.pruningInstance) return host.pruningInstance;
        const names = ["source", "run", "reconnect", "cancel", "status", "work", "chart", "slider", "count", "percent", "summary", "coverage", "cache", "review", "page", "keep", "export", "exportStatus", "download", "split", "splitPolicy", "exportPreview", "mode", "sceneComponents", "sceneCard", "objectCard", "objectRecipe", "objectAdjustment", "objectAdvanced", "openDQE", "inheritance", "components", "sizeDiagnostics", "recipeSummary", "changeImpact", "draftBadge", "activeRecipe", "objectWeight", "weightLabel", "objectThreshold", "sceneThreshold", "indexThreshold", "sceneRequirement", "negativeThreshold", "rarityScale", "neighborK", "scopeHelp", "rulePreset", "scaleProtection", "bulkCount", "bulkScope", "bulkPreview", "bulkUndo", "search", "filter", "sort", "sentinel", "loadMore"];
        names.push("policyStrict", "policyBudget", "policyHelp", "planBadge", "planObjectWeight", "planWeightLabel", "groupControl", "weightControl", "lossGroupRange", "lossGroupCap", "lossPercentRange", "lossPercentCap", "planStatus", "planProgress", "applyPlan", "discardPlan", "recalculatePlan", "lossSummary", "classWarning", "impactView", "familyView", "impactHelp", "familyToolbar", "impact");
        const el = Object.fromEntries(names.map(k => [k, host.querySelector(`[data-pruning="${k}"]`)]));
        const state = { context: null, jobId: "", result: null, capabilities: null, profiles: null, plan: null, review: null, removed: 0, next: null, offset: 0, serial: 0, operation: 0, controller: null, busy: false, exporting: false, reviewPending: false, checkpointing: false, externalRefreshing: false, recoveryRequired: false, pending: null, exportId: "", exportBinding: null, preview: null, timer: null, familyTimer: null, initialized: false, families: new Map(), familyNodes: new Map(), familyDrafts: new Map(), noteDrafts: new Map(), listSignature: "", sceneRows: [], objectRows: [], preparation: {}, generation: "", lastBatch: "", dialog: null, dialogContext: null, calibrationProfile: null, calibrationDraft: "", savedLab: null };
        state.planning = { active: null, draft: null, preview: null, dirty: false, pending: false, sequence: 0, timer: null, error: "", view: "impact", rows: [], next: null, listSequence: 0, loading: false };
        let recipeLab = null;
        const current = c => state.context === c && signature(cleanContext(adapters.context())) === signature(c);
        const storageKey = c => `tator.pruning.v2:${signature(c)}`;
        const outboxScope = c => digest([c?.apiRoot, c?.backendId]);
        const outboxPrefix = c => `tator.pruning.operations.v1:${outboxScope(c)}:`;
        function unresolved(context) {
            const prefix = outboxPrefix(context), values = [];
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (!key?.startsWith(prefix)) continue;
                const record = JSON.parse(localStorage.getItem(key));
                if (record?.status === "pending") values.push(record.pending);
            }
            return values.sort((a, b) => a.value.request_id.localeCompare(b.value.request_id));
        }
        function saveOperation(pending, status, receipt = null) {
            const key = `${outboxPrefix(state.context)}${pending.value.request_id}`;
            const prior = JSON.parse(localStorage.getItem(key) || "null");
            if (prior && digest(prior.pending) !== digest(pending)) throw new Error("An operation recovery record changed. Preserve it and reconnect before editing.");
            if (prior && prior.status !== "pending" && status === "pending") return;
            localStorage.setItem(key, digest({ pending, status, receipt }));
        }
        const post = value => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) });
        const payload = () => ({ removed_count: state.removed, selection_revision: state.result.selection_revision });
        const reviewPayload = () => ({ ...payload(), expected_review_revision: state.review?.review_revision ?? state.plan?.review_revision ?? state.result?.review_revision ?? 0, plan_revision: state.review?.plan_revision ?? state.plan?.plan_revision ?? state.result?.plan_revision ?? 0, expected_plan_revision: state.review?.plan_revision ?? state.plan?.plan_revision ?? state.result?.plan_revision ?? 0 });
        const maxRemoved = () => state.result?.max_removed_count ?? 0;
        const clamp = v => Math.min(state.result?.image_count ?? 0, Math.max(0, Math.round(Number(v) || 0)));
        const active = () => document.getElementById("tabDatasetPruning")?.classList.contains("active");
        const heads = () => state.capabilities?.heads || [];
        const headScope = head => head.representation_scope || head.scope || "whole_image";
        const headLabel = head => state.profiles?.profiles?.[head.id]?.display_label || head.display_label || head.label || head.id;
        const headArchived = head => Boolean(state.profiles?.profiles?.[head.id]?.archived || head.archived);
        const className = id => state.result?.class_names?.[id] || `Class ${id}`;
        function message(text, error = false) { el.status.textContent = text; el.status.classList.toggle("pruning-error", error); }
        const planningDirty = () => state.planning.dirty;
        const finiteNumber = value => typeof value === "number" && Number.isFinite(value);
        const measuredCount = value => finiteNumber(value) ? number(value) : "Not measured";
        const measuredPercent = value => finiteNumber(value) ? pct(value) : "Not measured";
        function normalizeSettings(value = {}) {
            const config = state.result?.resolved_config || state.result?.config || {};
            return { policy: value.policy === "budgeted_coverage" ? "budgeted_coverage" : "strict_duplicates", object_weight: Number(value.object_weight ?? config.object_weight ?? .9), max_removed_images: clamp(value.max_removed_images ?? state.removed), max_affected_appearance_groups: Math.max(0, Math.round(Number(value.max_affected_appearance_groups) || 0)), max_lost_appearance_weight_fraction: Math.min(1, Math.max(0, Number(value.max_lost_appearance_weight_fraction) || 0)) };
        }
        function syncPlanning(force = false) {
            const p = state.planning;
            p.active = normalizeSettings(state.review?.active_settings || state.result?.active_settings || state.plan?.active_settings || {});
            if (force || !p.dirty) p.draft = { ...p.active };
        }
        function appliedPlan() {
            return { ...(state.plan || {}), ...(state.result?.plan || {}), ...(state.result?.active_plan || {}), ...(state.review?.plan || {}), ...(state.review?.active_plan || {}), ...Object.fromEntries(["proposed_loss", "approved_loss", "appearance_group_count", "total_appearance_weight"].filter(key => state.review?.[key] !== undefined).map(key => [key, state.review[key]])), accounting_outdated: state.review?.accounting_outdated ?? state.result?.accounting_outdated ?? state.plan?.accounting_outdated ?? false };
        }
        function displayPlan() { return state.planning.preview?.status === "ready" ? state.planning.preview.plan : { ...appliedPlan(), ...(planningDirty() ? { proposed_loss: undefined } : {}) }; }
        function cancelSettingsPreview() {
            const p = state.planning, preview = p.preview;
            p.sequence++; p.listSequence++; clearTimeout(p.timer); p.timer = null; p.pending = false; p.preview = null;
            if (preview?.preview_id && preview.context && preview.jobId) request(`/jobs/${encodeURIComponent(preview.jobId)}/plan/previews/${encodeURIComponent(preview.preview_id)}/cancel`, post({}), preview.context).catch(() => {});
        }
        function resetPlanning() {
            cancelSettingsPreview();
            Object.assign(state.planning, { active: null, draft: null, dirty: false, error: "", rows: [], next: null, loading: false });
            el.impact.replaceChildren();
        }
        function renderLossSummary(plan) {
            const p = state.planning;
            const outdated = appliedPlan().accounting_outdated, staleProposal = outdated && p.preview?.status !== "ready";
            el.lossSummary.replaceChildren();
            for (const [label, loss] of [[staleProposal ? "Previous plan · coverage needs updating" : p.dirty ? "Preview · all proposed removals" : "Applied plan · all proposals", staleProposal ? null : plan.proposed_loss], [outdated ? "Approvals need an updated coverage check" : "Applied approved subset · actual export", outdated ? null : appliedPlan().approved_loss]]) {
                const card = h("div", "pruning-loss-card"); card.append(h("h4", "", label));
                const values = h("div", "pruning-loss-values");
                const measured = loss?.measured !== false;
                for (const [value, title] of [[measuredCount(measured ? loss?.affected_appearance_groups : null), "Affected appearance groups"], [measuredPercent(measured ? loss?.lost_appearance_weight_fraction : null), "Rarity-weighted loss"], [measuredCount(measured ? loss?.uncovered_objects : null), "Original objects without a retained match"], [measuredCount(measured ? loss?.removed_annotations : null), "Annotations in removed images"]]) { const item = h("div"); item.append(h("strong", "", value), h("span", "", title)); values.append(item); }
                card.append(values); el.lossSummary.append(card);
            }
            const disappearing = staleProposal ? [] : plan.proposed_loss?.disappearing_classes || [];
            const approved = outdated ? [] : appliedPlan().approved_loss?.disappearing_classes || [];
            const describe = rows => rows.map(row => typeof row === "object" ? row.class_name || className(row.class_id) : className(row)).join(", ");
            el.classWarning.hidden = !disappearing.length && !approved.length;
            el.classWarning.textContent = `${disappearing.length ? `No retained examples of these classes in the ${p.dirty ? "preview" : "proposed plan"}: ${describe(disappearing)}. ` : ""}${approved.length ? `Approved export loses these classes: ${describe(approved)}. ` : ""}Class disappearance is allowed by your budgets. Keep an image to preserve an example; there is no automatic class floor.`;
        }
        function previewProgressLabel(preview) {
            const message = String(preview?.message || ""), phase = message.replace(/_resumed$/, ""), resumed = phase !== message;
            const classPhase = phase.match(/^(neighbors|object_ranking|rarity_neighbors|rarity_reusing_frozen_neighbors|exact_rarity)_class_(\d+)$/);
            if (classPhase) {
                const labels = { neighbors: "Finding similar cached objects", object_ranking: "Ranking cached object matches", rarity_neighbors: "Comparing appearance groups", rarity_reusing_frozen_neighbors: "Reusing saved appearance comparisons", exact_rarity: resumed ? "Reusing measured appearance rarity" : "Measuring appearance rarity" };
                return `${labels[classPhase[1]]} · ${className(classPhase[2])}`;
            }
            if (phase === "groups") return resumed ? "Reusing saved appearance groups" : "Grouping similar cached object appearances";
            if (phase === "scene_ranking") return "Ranking cached whole-image matches";
            if (phase === "verify_objects") return "Checking direct retained object matches";
            if (phase === "plan") return "Planning removals within your limits";
            const pairs = message.match(/^Preparing compact cached pair measurements: (\d+)$/);
            if (pairs) return `Preparing cached comparisons · ${number(Number(pairs[1]))} pairs read`;
            if (message === "Checking the frozen source and planning settings") return "Checking the saved source and planning settings";
            if (message === "Replaying cached object fingerprints with saved size correction") return "Preparing cached object features with saved size correction";
            return message || "Preparing cached features and coverage";
        }
        function renderPlanning() {
            if (!state.result) return;
            const p = state.planning; if (!p.active) syncPlanning();
            const settings = p.draft || p.active, plan = displayPlan(), budgeted = settings.policy === "budgeted_coverage";
            const needsCoverageUpdate = appliedPlan().accounting_outdated && !p.pending && p.preview?.status !== "ready";
            const groupCount = plan.appearance_group_count ?? plan.metrics?.appearance_group_count ?? plan.accounting?.appearance_group_count ?? state.result.appearance_group_count ?? state.result.accounting?.appearance_group_count;
            el.policyStrict.setAttribute("aria-pressed", String(!budgeted)); el.policyBudget.setAttribute("aria-pressed", String(budgeted));
            el.policyHelp.textContent = budgeted ? "Objects can be covered by different retained images, or lose coverage within both budgets. Every removed image still needs a directly matching retained scene. No class has an automatic minimum." : "Strict duplicates use the existing one-to-one object and scene rules. Loss-budget controls apply to the coverage policy; changing the object weight does not remove matching gates.";
            el.planObjectWeight.value = settings.object_weight; el.planWeightLabel.textContent = `${pct(settings.object_weight)} objects · ${pct(1 - settings.object_weight)} scene`;
            el.slider.value = el.count.value = settings.max_removed_images; el.percent.value = state.result.image_count ? (100 * settings.max_removed_images / state.result.image_count).toFixed(2) : "0";
            for (const node of [el.lossGroupRange, el.lossGroupCap]) { node.max = finiteNumber(groupCount) ? groupCount : 1000000000; node.value = settings.max_affected_appearance_groups; }
            el.lossGroupRange.hidden = !finiteNumber(groupCount);
            el.lossPercentRange.value = el.lossPercentCap.value = (100 * settings.max_lost_appearance_weight_fraction).toFixed(2);
            el.groupControl.dataset.inactive = el.weightControl.dataset.inactive = String(!budgeted);
            el.planBadge.textContent = needsCoverageUpdate ? "Coverage check needs updating" : p.pending ? "Computing preview" : p.dirty ? p.preview?.status === "ready" ? "Preview · not applied" : "Unapplied changes" : "Applied plan";
            el.planBadge.classList.toggle("warning", p.dirty || needsCoverageUpdate);
            el.planProgress.hidden = !p.pending;
            if (finiteNumber(p.preview?.progress)) el.planProgress.value = p.preview.progress; else el.planProgress.removeAttribute("value");
            el.planProgress.setAttribute("aria-label", `${previewProgressLabel(p.preview)}. Current stage only; progress can restart between stages.`);
            el.planStatus.textContent = needsCoverageUpdate ? `${p.error ? `${p.error} ` : ""}Coverage check needs updating. Recalculate proposals, then Apply and review them again. Saved features and Keep choices are reused.` : p.error || (p.pending ? `${previewProgressLabel(p.preview)} · progress is per stage` : p.dirty ? p.preview?.status === "ready" ? "Preview ready. Inspect its losses, then Apply or Discard. No approvals have changed." : "Settings changed. Preparing a new preview …" : "Applied settings. Keep choices leave room below the ceiling until you explicitly recalculate.");
            el.planStatus.classList.toggle("pruning-error", !!p.error);
            el.impactView.setAttribute("aria-pressed", String(p.view === "impact")); el.familyView.setAttribute("aria-pressed", String(p.view === "family"));
            el.impact.hidden = p.view !== "impact"; el.review.hidden = p.view !== "family"; el.familyToolbar.hidden = p.view !== "family"; el.impactHelp.hidden = p.view !== "impact";
            el.impactHelp.textContent = budgeted ? "Highest restoration value first: keeping an image may restore a whole appearance group or only some of its objects. Keeping an image can also restore matches for objects in other proposed removals. Proposed losses and actual approved-export losses are shown separately." : "Inspect the strict duplicate suggestions and their retained witnesses. Restoration and appearance-loss accounting are shown only when measured; strict matching is not a claim of zero global diversity loss.";
            el.review.classList.toggle("pruning-preview-muted", p.dirty);
            if (p.view === "impact") { el.loadMore.hidden = p.next === null; el.page.textContent = `${number(p.rows.length)} impact cards loaded${p.dirty ? " · preview" : " · applied plan"}`; el.sentinel.textContent = p.loading ? "Loading impact cards …" : p.next === null ? "All available impact cards loaded." : "More impact cards available."; }
            else if (p.dirty && p.preview?.status !== "ready") el.page.textContent = "Preview pending · previous families shown read-only";
            renderLossSummary(plan);
        }
        function editSettings(changes = {}, force = false) {
            if (!state.result || state.busy || state.exporting || state.externalRefreshing || state.pending || state.recoveryRequired) return;
            const p = state.planning; if (!p.active) syncPlanning();
            const next = normalizeSettings({ ...p.draft, ...changes });
            cancelSettingsPreview(); state.serial++; state.reviewPending = false; el.review.setAttribute("aria-busy", "false"); p.draft = next; p.dirty = force || digest(next) !== digest(p.active); p.error = "";
            invalidatePreview();
            p.rows = []; p.next = null; el.impact.replaceChildren(h("p", "pruning-help", p.dirty ? "Impact cards will follow the verified preview." : "Loading the applied plan …"));
            render();
            if (!p.dirty) { (p.view === "impact" ? impactCards(state.context, true) : families(state.context, true)).catch(e => handleError(e)); return; }
            p.pending = true; renderPlanning(); controls();
            p.timer = setTimeout(() => previewSettings(), 200);
        }
        async function previewSettings() {
            const p = state.planning;
            if (!p.dirty || !state.result) return;
            if (state.busy || state.exporting || state.reviewPending || state.externalRefreshing || state.pending || state.recoveryRequired) { p.timer = setTimeout(previewSettings, 200); return; }
            const ticket = p.sequence, context = state.context, jobId = state.jobId, binding = reviewPayload(), settings = { ...p.draft };
            const matches = () => current(context) && jobId === state.jobId && ticket === p.sequence && digest(settings) === digest(p.draft) && binding.selection_revision === state.result?.selection_revision && binding.expected_review_revision === reviewPayload().expected_review_revision && binding.expected_plan_revision === reviewPayload().expected_plan_revision;
            p.pending = true; p.error = ""; renderPlanning(); controls();
            try {
                let value = await request(`/jobs/${encodeURIComponent(jobId)}/plan/preview`, post({ kind: "settings", selection_revision: binding.selection_revision, expected_review_revision: binding.expected_review_revision, expected_plan_revision: binding.expected_plan_revision, settings }), context);
                if (!matches()) { if (value.preview_id) request(`/jobs/${encodeURIComponent(jobId)}/plan/previews/${encodeURIComponent(value.preview_id)}/cancel`, post({}), context).catch(() => {}); return; }
                while (matches()) {
                    p.preview = { ...value, context, jobId, binding, settings }; renderPlanning();
                    if (value.status === "ready") {
                        if (!value.plan || !value.preview_digest) throw new Error("The preview has no verified plan binding. Recalculate before applying.");
                        p.pending = false; render(); controls();
                        if (p.view === "impact") await impactCards(context, true); else await families(context, true);
                        return;
                    }
                    if (!["queued", "running"].includes(value.status)) throw new Error(value.message || `Preview ${value.status || "failed"}`);
                    await pause(400); if (!matches()) return;
                    value = await request(`/jobs/${encodeURIComponent(jobId)}/plan/previews/${encodeURIComponent(value.preview_id)}`, {}, context);
                }
            } catch (error) {
                if (matches() && error.name !== "AbortError") { p.error = error.message; p.pending = false; p.preview = null; renderPlanning(); controls(); }
            }
        }
        async function discardSettings() {
            cancelSettingsPreview(); state.serial++; state.reviewPending = false; el.review.setAttribute("aria-busy", "false"); const p = state.planning; p.dirty = false; p.draft = { ...p.active }; p.error = ""; p.rows = []; p.next = null; render();
            if (p.view === "impact") await impactCards(state.context, true); else await families(state.context, true);
        }
        async function applySettings() {
            const p = state.planning, preview = p.preview;
            if (!p.dirty || p.pending || preview?.status !== "ready" || !preview.preview_id || digest(preview.settings) !== digest(p.draft)) return;
            await reviewAction({ action: "apply_plan", preview_id: preview.preview_id, preview_digest: preview.preview_digest });
        }
        function remember() {
            if (!state.context) return false;
            try {
                if (state.pending) saveOperation(state.pending, "pending");
                const value = { jobId: state.jobId, removed: state.removed, exportId: state.exportId, exportBinding: state.exportBinding, pending: state.pending, lastBatch: state.lastBatch, draft: config(), recipeLab: recipeLab?.snapshot() || state.savedLab, reviewView: state.planning.view, filter: el.filter.value, query: el.search.value, sort: el.sort.value, scroll: window.scrollY, noteDrafts: Array.from(state.noteDrafts), familyDrafts: Array.from(state.familyDrafts, ([id, v]) => [id, { count: v.count, keeperIds: v.keeperIds }]) };
                localStorage.setItem(storageKey(state.context), digest(value)); return true;
            } catch (_) { return false; }
        }
        function invalidateExport() { state.exportId = ""; state.exportBinding = null; el.download.hidden = true; el.exportStatus.textContent = ""; }
        function invalidatePreview(text = "The preview changed. Generate a fresh preview before applying it.") {
            state.preview = null;
            if (state.dialog?.dataset.kind === "preview") { const commit = state.dialog.querySelector('[data-preview-commit]'); if (commit) commit.disabled = true; const status = state.dialog.querySelector('[data-preview-status]'); if (status) status.textContent = text; }
        }
        function controls() {
            const ready = !!state.result && state.context && current(state.context);
            const blocked = state.busy || state.exporting || state.externalRefreshing || state.recoveryRequired || !!state.pending;
            const settled = ready && !blocked && !state.reviewPending && !planningDirty();
            el.run.disabled = blocked || !state.capabilities || !state.context?.annotationSource?.id || state.context?.inventory?.image_count === 0 || state.context?.imageCount === 0;
            el.cancel.hidden = !state.busy || !state.jobId;
            el.reconnect.hidden = !state.pending && !state.recoveryRequired;
            el.reconnect.disabled = state.busy || state.exporting;
            el.work.hidden = !state.result;
            for (const k of ["slider", "count", "percent"]) el[k].disabled = !ready || blocked;
            const budgeted = state.planning.draft?.policy === "budgeted_coverage";
            const configMode = state.result?.resolved_config?.evidence_mode || state.result?.config?.evidence_mode || "compound";
            for (const k of ["policyStrict", "policyBudget", "planObjectWeight", "recalculatePlan"]) el[k].disabled = !ready || blocked;
            el.policyBudget.disabled = !ready || blocked || configMode !== "compound";
            el.planObjectWeight.disabled = !ready || blocked || configMode !== "compound";
            for (const k of ["lossGroupRange", "lossGroupCap", "lossPercentRange", "lossPercentCap"]) el[k].disabled = !ready || blocked || !budgeted;
            el.applyPlan.disabled = !ready || blocked || state.reviewPending || !planningDirty() || state.planning.pending || state.planning.preview?.status !== "ready";
            el.discardPlan.disabled = !ready || blocked || !planningDirty();
            for (const k of ["impactView", "familyView"]) el[k].disabled = !ready || blocked;
            for (const k of ["bulkPreview", "bulkCount", "bulkScope", "export", "splitPolicy"]) el[k].disabled = !settled;
            el.export.disabled = !settled || !(state.plan?.approved_count ?? state.review?.approved_count ?? 0);
            el.bulkUndo.hidden = !state.lastBatch; el.bulkUndo.disabled = !settled;
            host.querySelectorAll('[data-review-action]').forEach(n => { n.disabled = !settled || n.dataset.requiresConfirmation === "true"; });
            host.querySelectorAll('[data-family-count]').forEach(n => { n.disabled = !settled; });
            el.loadMore.disabled = blocked || state.reviewPending || state.planning.loading || (planningDirty() && state.planning.preview?.status !== "ready"); el.loadMore.hidden = state.planning.view === "impact" ? state.planning.next === null : state.next === null;
            host.querySelectorAll('[data-recipe-control]').forEach(n => { n.disabled = blocked || n.dataset.unsupported === "true"; });
            el.mode.disabled = el.objectRecipe.disabled = el.objectAdjustment.disabled = el.rulePreset.disabled = blocked;
            recipeLab?.sync();
            if (state.dialog) state.dialog.querySelectorAll('[data-mutation]').forEach(n => { n.disabled = blocked; });
        }
        async function request(path, options = {}, context = state.context) {
            if (!context?.apiRoot) throw new Error("Connect a backend first.");
            const signal = state.controller?.signal;
            const response = await fetch(`${context.apiRoot}/dataset_pruning${path}`, { ...options, signal, headers: { ...(options.headers || {}), ...(context.backendId ? { "X-Tator-Backend-Id": context.backendId } : {}) } });
            const body = await response.json().catch(() => ({}));
            if (!current(context)) throw new DOMException("The dataset or backend changed.", "AbortError");
            if (!response.ok) { const error = new Error(typeof body.detail === "string" ? body.detail : body.detail?.message || body.message || `Pruning request failed (${response.status}).`); error.status = response.status; error.detail = body.detail; throw error; }
            return body;
        }
        async function mutation(kind, path, value, context, resume = false) {
            if (!resume) {
                const foreign = unresolved(context)[0];
                if (foreign) { state.pending = foreign; state.recoveryRequired = true; controls(); throw new Error("Another tab has an unresolved operation for this dataset. Reconnect to recover its exact receipt before editing."); }
                if (state.pending) throw new Error("Reconnect the outstanding request before making another change.");
                state.pending = { kind, path, value: { ...value, request_id: nonce() }, context_signature: signature(context), source_scope: outboxScope(context), annotation_scope: sourceKey(context), target_identity: kind === "profile" ? value.source_identity : sourceKey(context), jobId: state.jobId };
                if (!remember()) { state.pending = null; throw new Error("Browser storage could not save request recovery. Restore browser storage before retrying."); }
            }
            const pending = state.pending;
            if (!pending || (pending.source_scope ? pending.source_scope !== outboxScope(context) : pending.context_signature !== signature(context))) throw new Error("This outstanding request belongs to another source. Reopen that source to recover it.");
            if (pending.kind !== "profile" && pending.annotation_scope && pending.annotation_scope !== sourceKey(context)) throw new Error(`An unresolved operation belongs to ${pending.annotation_scope}. Reopen that dataset to recover its exact receipt.`);
            for (let attempt = 0; attempt < 2; attempt++) {
                try { const answer = await request(pending.path, post(pending.value), context); if (["review", "profile", "calibration"].includes(pending.kind) && answer.outcome !== "committed") throw new Error("The backend did not return a durable commit receipt. Reconnect to recover this operation."); saveOperation(pending, "committed", answer.receipt || null); state.pending = unresolved(context)[0] || null; state.recoveryRequired = !!state.pending; remember(); return answer; }
                catch (error) {
                    if (!current(context) || error.name === "AbortError") throw error;
                    if (error.status && error.status < 500 && ![408, 429].includes(error.status)) { saveOperation(pending, "rejected", { status: error.status, message: error.message }); state.pending = unresolved(context)[0] || null; remember(); throw error; }
                    if (attempt === 1) { state.recoveryRequired = true; controls(); remember(); throw new Error("The request outcome is unknown. Reconnect to recover the same saved operation."); }
                }
            }
        }
        function begin(raw) {
            const context = cleanContext(raw);
            if (state.context && signature(state.context) === signature(context)) return state.context;
            remember(); resetPlanning(); state.operation++; state.serial++; state.controller?.abort(); state.controller = new AbortController();
            Object.assign(state, { context, jobId: "", result: null, plan: null, review: null, removed: 0, next: null, offset: 0, busy: false, exporting: false, reviewPending: false, externalRefreshing: false, externalRefreshNeeded: false, recoveryRequired: false, pending: null, exportId: "", exportBinding: null, capabilities: null, profiles: null, preview: null, generation: "", lastBatch: "", initialized: false, calibrationProfile: null, calibrationDraft: "", savedLab: null });
            recipeLab?.reset();
            for (const value of state.familyDrafts.values()) clearTimeout(value.timer); state.families.clear(); state.familyNodes.clear(); state.familyDrafts.clear(); state.noteDrafts.clear(); state.listSignature = ""; el.review.replaceChildren();
            clearTimeout(state.timer); clearTimeout(state.familyTimer); el.download.hidden = true;
            if (state.dialog?.open) state.dialog.close(); controls(); renderSource(); return context;
        }
        function renderSource() {
            const c = state.context || cleanContext(adapters.context());
            const inventory = c.inventory || {}, count = inventory.image_count ?? c.imageCount ?? state.result?.image_count;
            const connection = state.capabilities ? `Connected to ${c.backendName || c.backendId || c.apiRoot}` : c.apiRoot ? "Backend connection not yet verified" : "No backend connected";
            el.source.replaceChildren();
            if (!c.annotationSource?.id) {
                el.source.append(h("span", "", `No dataset attached · ${connection}. Open a dataset to analyze its saved images and labels.`));
                const actions = h("div", "pruning-inline"); actions.append(button("Open Dataset Management", () => document.getElementById("tabDatasetsButton")?.click()), button("Open Label Images", () => document.getElementById("tabLabelingButton")?.click()), button("Backend Config", () => document.getElementById("tabSettingsButton")?.click())); el.source.append(actions); return;
            }
            const name = c.datasetName || c.annotationSource.label || `${c.annotationSource.kind}: ${c.annotationSource.id}`;
            el.source.append(h("span", "", `${name} · ${connection} · ${count === undefined ? "image inventory awaits analysis validation" : `${number(count)} images`} · saved revision ${c.revision ?? "not saved"}${c.analysisJobId ? ` · DQE ${c.analysisJobId}` : " · selected object features use the shared DQE producer"}`));
            if (count === 0) el.source.append(h("p", "pruning-help", "This source has no images. Open or add dataset images before analyzing."));
        }
        function modelChoices(encoder) {
            const caps = state.capabilities || {};
            const listed = caps.scene_models?.[encoder] || caps.models?.[encoder] || (encoder === "cradio" ? caps.cradio_models : caps.dinov3_models) || [];
            const values = listed.map(v => typeof v === "string" ? [v, v] : [v.id || v.model, v.label || v.id || v.model]);
            for (const head of heads().filter(v => v.encoder_type === encoder)) if (!values.some(([id]) => id === head.encoder_model)) values.push([head.encoder_model, head.encoder_model]);
            const recipe = adapters.getRecipe?.() || {};
            const fallback = encoder === "dinov3" ? recipe.encoder_model || caps.default_dinov3_model || "facebook/dinov3-vitb16-pretrain-lvd1689m" : caps.default_cradio_model;
            if (fallback && !values.some(([id]) => id === fallback)) values.push([fallback, fallback]);
            return values;
        }
        function sceneControls(components) {
            state.sceneRows = []; el.sceneComponents.replaceChildren();
            for (const encoder of ["dinov3", "cradio"]) {
                const existing = (components || []).find(v => (v.encoder_type || v.encoder || "").includes(encoder) || heads().find(q => q.id === v.head_id)?.encoder_type === encoder);
                const models = modelChoices(encoder), row = h("div", "pruning-component-row");
                const enabled = input("checkbox", "", { checked: existing ? existing.enabled !== false : encoder === "dinov3" });
                const title = h("div", "pruning-component-title"); title.append(field(encoder === "dinov3" ? "DINOv3" : "C-RADIO", enabled));
                const model = select(models, existing?.model || existing?.encoder_model || models[0]?.[0]);
                const aggregation = select([["pooled", "Pooled backbone"], ["local_salad", "Trained SALAD"]], existing?.head_id ? "local_salad" : "pooled");
                const head = select([], existing?.head_id), weight = input("number", existing?.weight ?? 1, { min: "0", max: "1", step: "0.05" });
                const pooling = select(encoder === "dinov3" ? [["pooler", "DINO pooler"], ["cls", "CLS"], ["patch_mean", "Patch mean"], ["cls_patch_concat", "CLS + patch mean"]] : [["summary", "Native summary"], ["spatial_mean", "Spatial mean"], ["summary_spatial_concat", "Summary + spatial mean"]], existing?.pooling || (encoder === "dinov3" ? "pooler" : "summary"));
                const adjustment = select([["raw", "Raw features"], ["remove", "Image-size correction (shared fitter)"]], existing?.adjustment === "none" ? "raw" : existing?.adjustment || "raw");
                const inputSize = encoder === "cradio" ? select([["", "Independent native grid"], ["512", "512 square · controlled report recipe"], ["768", "768 square"], ["1024", "1024 square"]], existing?.input_size ?? "") : null;
                const imageCorrection = state.capabilities?.scene_adjustments?.includes("remove");
                if (!imageCorrection) adjustment.options[1].disabled = true;
                const scope = h("p", "pruning-help");
                const item = { encoder, enabled, model, aggregation, head, weight, pooling, adjustment, inputSize, scope, adjustmentStrength: existing?.adjustment_strength };
                adjustment.addEventListener("change", () => { item.adjustmentStrength = adjustment.value === "remove" ? 1 : 0; });
                function compatibleHeads() {
                    const selected = head.value || existing?.head_id;
                    head.replaceChildren(new Option("Select a compatible trained head", ""));
                    for (const v of heads().filter(v => v.encoder_type === encoder && (!model.value || v.encoder_model === model.value) && headScope(v) === "whole_image" && !headArchived(v))) head.add(new Option(headLabel(v), v.id));
                    if (Array.from(head.options).some(v => v.value === selected)) head.value = selected;
                    head.hidden = aggregation.value !== "local_salad"; if (head.parentElement) head.parentElement.hidden = aggregation.value !== "local_salad"; if (pooling.parentElement) pooling.parentElement.hidden = aggregation.value === "local_salad"; pooling.dataset.unsupported = String(aggregation.value === "local_salad"); pooling.disabled = aggregation.value === "local_salad";
                    const chosen = heads().find(v => v.id === head.value);
                    scope.textContent = chosen ? `Head ${chosen.id} · trained on ${number(chosen.train_image_count)} historical images · ${chosen.descriptor_dim || chosen.descriptor_dimension || "saved"}D · ${chosen.reference_label || "recorded training reference"}` : aggregation.value === "local_salad" ? "Select a head or explicitly train one in the existing ingestion profile tool. SAM3 has no whole-image SALAD path." : "Existing pooled feature service; compatible cached vectors are reused.";
                }
                for (const n of [enabled, model, aggregation, head, weight, pooling, adjustment, inputSize].filter(Boolean)) { n.dataset.recipeControl = ""; n.addEventListener("change", () => { compatibleHeads(); draftChanged(); }); }
                if (!models.length) { enabled.checked = false; enabled.dataset.unsupported = "true"; enabled.disabled = true; }
                row.append(title, field("Model", model), field("Aggregation", aggregation), field("SALAD profile", head), field("Backbone fusion weight", weight), field("Pooled representation", pooling), field("Size policy", adjustment), scope); compatibleHeads(); if (!models.length) scope.textContent = "No compatible model is available on this backend."; el.sceneComponents.append(row); state.sceneRows.push(item);
                if (inputSize) row.insertBefore(field("Whole-image encoder grid", inputSize, "An explicit square uses BICUBIC direct stretching.512 is the controlled report variant; native remains the default. Object crops keep their own336 preparation."), scope);
            }
        }
        function objectControls(configValue = {}) {
            state.objectRows = []; el.objectAdvanced.replaceChildren();
            const selectedPreset = state.capabilities?.object_recipes?.find(v => v.id === (configValue.object_recipe || el.objectRecipe.value));
            const inherited = state.capabilities?.inherited_recipe || {};
            const currentRecipe = adapters.getRecipe?.();
            const recipe = selectedPreset?.recipe_config || (currentRecipe && Object.keys(currentRecipe).length ? currentRecipe : inherited.recipe_config) || {};
            const components = configValue.object_components || selectedPreset?.bbox?.components || inherited.bbox?.components || [];
            const grid = h("div", "pruning-rule-grid");
            for (const [id, label, fallback] of [["dinov3", "DINOv3", .50625], ["sam3", "SAM3 mask features", .16875], ["salad", "DINO crop SALAD", .075], ["cradio", "C-RADIO", .25]]) {
                const existing = components.find(v => v.id === id || v.component === id || (id !== "salad" && v.encoder === id)) || {};
                const activeWeight = existing.weight ?? (components.length ? 0 : fallback);
                const weight = input("number", activeWeight, { min: "0", max: "1", step: "0.025" });
                const enabled = input("checkbox", "", { checked: existing.enabled !== false && activeWeight > 0 });
                const pooling = select(id === "cradio" ? [["summary", "Native summary"], ["spatial_mean", "Spatial mean"], ["summary_spatial_concat", "Summary + spatial mean"]] : id === "dinov3" ? [["pooler", "DINO pooler"], ["cls", "CLS"], ["patch_mean", "Patch mean"], ["cls_patch_concat", "CLS + patch mean"]] : [["inherited", "Existing feature aggregation"]], (["dinov3", "cradio"].includes(id) ? existing.pooling : null) || (id === "cradio" ? recipe.cradio_pooling || "summary" : id === "dinov3" ? recipe.dinov3_pooling || recipe.embedding_pooling || "pooler" : "inherited"));
                const card = h("div", "pruning-component-row"), model = ["dinov3", "cradio"].includes(id) ? select(modelChoices(id), existing.model || existing.encoder_model || (id === "cradio" ? recipe.quality_cradio_model : recipe.encoder_model) || modelChoices(id)[0]?.[0]) : null; card.append(field(label, enabled), field("Final component weight", weight), field("Pooling", pooling)); if (model) { model.dataset.recipeControl = ""; model.addEventListener("change", draftChanged); card.append(field("Backbone model", model)); }
                const inheritedStrength = existing.adjustment_strength ?? (existing.adjustment === "none" || existing.adjustment === "raw" ? 0 : Object.keys(existing).length ? 1 : id === "cradio" ? 0 : 1);
                const policy = configValue.object_adjustment || el.objectAdjustment.value;
                const strength = input("number", policy === "raw" ? 0 : policy === "remove_all" ? 1 : policy === "remove" ? (id === "cradio" ? 0 : 1) : inheritedStrength, { min: "0", max: "1", step: ".05" });
                strength.dataset.recipeControl = "";
                strength.addEventListener("change", () => { state.objectCustom = true; el.objectAdjustment.value = "inherit"; draftChanged(); });
                card.append(field("Size-and-shape correction strength", strength, "0 keeps the raw fingerprint; 1 subtracts the complete fitted geometry prediction, then normalizes. Intermediate values subtract that fraction before normalization."));
                const head = select([["", "Existing DQE head / selected recipe"]]);
                if (id === "salad") { for (const q of heads().filter(q => headScope(q) === "bbox_crop" && q.encoder_type === "dinov3" && !headArchived(q))) head.add(new Option(headLabel(q), q.id)); head.value = existing.head_id || ""; card.append(field("Crop SALAD profile", head)); }
                for (const n of [enabled, weight, pooling, head]) { n.dataset.recipeControl = ""; n.addEventListener("change", draftChanged); }
                state.objectRows.push({ id, enabled, weight, pooling, head, model, strength, inheritedStrength, sourceComponent: existing }); grid.append(card);
            }
            const compactWeight = input("number", recipe.compact_weight ?? .75, { min: "0", max: "1", step: ".05" });
            compactWeight.dataset.recipeControl = "";
            const compactComponents = state.capabilities?.object_recipes?.find(v => v.id === "precise_compact_v1")?.bbox?.components || [];
            const compactTotal = compactComponents.reduce((sum, v) => sum + Number(v.weight || 0), 0);
            compactWeight.disabled = !compactTotal;
            compactWeight.addEventListener("change", () => { state.objectCustom = true; const value = Math.min(1, Math.max(0, Number(compactWeight.value))); for (const row of state.objectRows) { const component = compactComponents.find(v => v.id === row.id); row.weight.value = (row.id === "cradio" ? 1 - value : value * Number(component?.weight || 0) / compactTotal).toFixed(5); } draftChanged(); });
            el.objectAdvanced.append(field("Existing Thorough compact / C-RADIO mix", compactWeight, "Uses the shared DQE compact preset's component ratios. Individual weights below remain editable."), grid);
            const preparation = configValue.object_preprocessing || selectedPreset?.bbox?.preprocessing || inherited.bbox?.preprocessing || {};
            const views = select([["single", "Standard crop"], ["tight_standard", "Tight + standard"], ["standard_context", "Standard + context"], ["tight_context", "Tight + context"]], preparation.embedding_view_mode || recipe.embedding_view_mode || "tight_context");
            const background = select([["full_crop", "Full crop"], ["mean_fill_outside_box", "Mean-color background"], ["blur_outside_box", "Blurred background"], ["darken_outside_box", "Darkened background"]], preparation.background_mode || recipe.background_mode || "full_crop");
            const padding = input("number", preparation.padding_ratio ?? recipe.padding_ratio ?? .08, { min: "0", max: "1", step: ".01" });
            const canonical = input("number", preparation.canonical_size ?? recipe.canonical_size ?? recipe.canonical_crop_size ?? 336, { min: "64", max: "1024", step: "16" });
            const crop = select([["padded_square", "Padded square"], ["padded", "Padded bbox"], ["tight", "Tight bbox"]], preparation.crop_mode || recipe.crop_mode || "padded_square");
            const preprocess = select([["canonical", "Canonical mean-color square + resize"], ["native", "Native crop preparation"]], preparation.preprocess_mode || recipe.preprocess_mode || "canonical");
            state.preparation = { views, background, padding, canonical, crop, preprocess };
            const prepGrid = h("div", "pruning-rule-grid"); prepGrid.append(field("Crop views", views), field("Background", background), field("Standard padding", padding, "Tight uses 0% padding; context uses at least 25%, or the standard padding when larger."), field("Prepared crop size", canonical, "Final encoder dimensions are reported separately."), field("Crop shape", crop), field("Preparation", preprocess));
            for (const n of Object.values(state.preparation)) { n.dataset.recipeControl = ""; n.addEventListener("change", draftChanged); }
            el.objectAdvanced.append(prepGrid);
        }
        function config() {
            const mode = el.mode.value;
            const sceneComponents = state.sceneRows.filter(v => v.enabled.checked).map(v => ({ encoder: v.aggregation.value === "local_salad" ? "local_salad" : `${v.encoder}_pooled`, encoder_type: v.encoder, model: v.model.value, head_id: v.aggregation.value === "local_salad" ? v.head.value : "", pooling: v.pooling.value, weight: Number(v.weight.value), adjustment: v.adjustment.value, ...(v.adjustmentStrength === undefined ? {} : { adjustment_strength: v.adjustmentStrength }), ...(v.inputSize?.value ? { input_size: Number(v.inputSize.value) } : {}) }));
            const value = { evidence_mode: mode, object_recipe: el.objectRecipe.value, object_adjustment: el.objectAdjustment.value, scene_components: mode === "objects_only" ? [] : sceneComponents, scene_threshold: Number(el.sceneThreshold.value), negative_scene_threshold: Number(el.negativeThreshold.value), object_threshold: Number(el.objectThreshold.value), index_threshold: Number(el.indexThreshold.value), object_weight: mode === "objects_only" ? 1 : mode === "scene_only" ? 0 : Number(el.objectWeight.value), rarity_scale_threshold: Number(el.rarityScale.value), neighbor_k: Number(el.neighborK.value), scale_protection: el.scaleProtection.value, scene_requirement: mode === "objects_only" ? "optional" : el.sceneRequirement.value, threshold_qualification: el.rulePreset.value === "demo" && el.objectRecipe.value === "demo_raw_dino_sam" && el.objectAdjustment.value === "raw" && !state.objectCustom && !state.preparationCustom && el.scaleProtection.value === "occupancy_legacy" && mode === "compound" && el.sceneRequirement.value === "required" && Object.entries({ sceneThreshold: .92, negativeThreshold: .94, objectThreshold: .80, indexThreshold: .87, objectWeight: .90, rarityScale: .60, neighborK: 32 }).every(([key, expected]) => Number(el[key].value) === expected) && sceneComponents.length === 1 && ["raw", "none"].includes(sceneComponents[0].adjustment) && sceneComponents[0].head_id === "WALDO3" && heads().find(v => v.id === "WALDO3")?.head_sha256 === "be870ed5e5704fd00594721cb17700d240e1eb212507788a1ce6f94611819ac5" ? "raw_demo_starting_settings" : "unmeasured_draft" };
            if (state.objectCustom) value.object_components = state.objectRows.map(v => ({ id: v.id, encoder: v.id === "salad" ? "dinov3" : v.id, enabled: v.enabled.checked, weight: Number(v.weight.value), adjustment: Number(v.strength.value) > 0 ? "remove_size_bias" : "none", adjustment_strength: Number(v.strength.value), ...(["dinov3", "cradio"].includes(v.id) ? { pooling: v.pooling.value, model: v.model?.value } : {}), ...(v.head.value ? { head_id: v.head.value, model: heads().find(q => q.id === v.head.value)?.encoder_model } : {}) }));
            if (state.preparationCustom && state.preparation.views) value.object_preprocessing = { embedding_view_mode: state.preparation.views.value, background_mode: state.preparation.background.value, padding_ratio: Number(state.preparation.padding.value), canonical_size: Number(state.preparation.canonical.value), crop_mode: state.preparation.crop.value, preprocess_mode: state.preparation.preprocess.value };
            if (state.calibrationProfile) { value.calibration_profile = state.calibrationProfile; value.threshold_qualification = "diagnostic_calibrated"; }
            return value;
        }
        const calibrationFingerprint = value => { const copy = { ...value }; delete copy.calibration_profile; delete copy.threshold_qualification; return digest(copy); };
        function applyConfig(value = {}) {
            const fields = { evidence_mode: "mode", object_recipe: "objectRecipe", object_adjustment: "objectAdjustment", scene_threshold: "sceneThreshold", negative_scene_threshold: "negativeThreshold", object_threshold: "objectThreshold", index_threshold: "indexThreshold", object_weight: "objectWeight", rarity_scale_threshold: "rarityScale", neighbor_k: "neighborK", scale_protection: "scaleProtection", scene_requirement: "sceneRequirement" };
            for (const [key, target] of Object.entries(fields)) if (value[key] !== undefined && el[target]) el[target].value = value[key];
            state.calibrationProfile = value.calibration_profile || null; state.calibrationDraft = "";
            state.objectCustom = !!value.object_components; state.preparationCustom = !!value.object_preprocessing;
            sceneControls(value.scene_components); objectControls(value);
            el.rulePreset.value = value.threshold_qualification === "raw_demo_starting_settings" ? "demo" : "draft";
            state.calibrationDraft = state.calibrationProfile ? calibrationFingerprint(config()) : "";
            draftChanged(false);
        }
        function draftChanged(save = true) {
            if (state.calibrationProfile && state.calibrationDraft && calibrationFingerprint(config()) !== state.calibrationDraft) { state.calibrationProfile = null; state.calibrationDraft = ""; }
            const value = config(), scope = value.evidence_mode;
            el.sceneCard.dataset.disabled = String(scope === "objects_only"); el.objectCard.dataset.disabled = String(scope === "scene_only");
            el.objectThreshold.disabled = scope === "scene_only"; el.sceneThreshold.disabled = scope === "objects_only" || value.scene_requirement !== "required";
            el.objectWeight.disabled = scope !== "compound";
            const weight = value.object_weight; el.weightLabel.textContent = `${pct(weight, 0)} objects · ${pct(1 - weight, 0)} scenes`;
            el.scopeHelp.textContent = scope === "objects_only" ? "Object appearance and same-class coverage are measured. Scene diversity is unmeasured; explicit negatives remain unless a compatible negative-scene rule is selected." : scope === "scene_only" ? "Whole-image similarity is measured. Object appearance and coverage are unmeasured. Label validity/conflicts still protect images; these proposals have no object-verified badge." : "Objects and scenes contribute separately. A zero contribution does not disable that branch's verification floor; disable the branch explicitly to remove its gate.";
            const scene = scope === "objects_only" ? "disabled" : value.scene_components.map(v => `${v.encoder_type === "dinov3" ? "DINOv3" : "C-RADIO"}${v.head_id ? ` + SALAD ${headLabel(heads().find(q => q.id === v.head_id) || { id: v.head_id })}` : " pooled"}`).join(" + ") || "select an enabled component";
            const objectOption = el.objectRecipe.selectedOptions[0]?.textContent || value.object_recipe;
            el.recipeSummary.textContent = `Draft · Scenes: ${scene} · Objects: ${scope === "scene_only" ? "disabled" : objectOption}${state.objectCustom && scope !== "scene_only" ? " (custom component settings)" : ""} · Size policy: ${scope === "scene_only" ? "image component policy" : el.objectAdjustment.selectedOptions[0]?.textContent}.`;
            el.draftBadge.textContent = state.result ? "Setup draft · active result below" : "Setup draft";
            const baseline = state.result?.resolved_config || state.result?.config;
            const changed = baseline && digest(value) !== digest(baseline);
            el.changeImpact.textContent = `${changed ? "Draft differs from the active analysis. " : ""}Keeper, budget and scene/object index-weight changes reuse compatible vectors. Component weights recompose object features and assignments. A new model/head/view computes only missing components; correction strength reuses a compatible fitted transform; only a changed fit population needs a new transform. Raw pixel features stay cached. ${value.threshold_qualification === "diagnostic_calibrated" ? "These cutoffs are bound to the complete saved recipe and source. Editing features or matching rules removes that calibration from the draft. The selected review does not establish population accuracy." : value.threshold_qualification === "raw_demo_starting_settings" ? "The raw demo gates belong to the named raw DINO/SAM recipe and its saved scene profile; they are starting settings, not a population accuracy guarantee." : "These thresholds are an unqualified draft for this resolved feature space. Inheriting a stronger or size-corrected recipe does not inherit the raw demo's calibration or removal count."}`;
            if (save) { invalidatePreview(); remember(); }
            controls();
        }
        function componentTable(manifest, recipe = null, cacheStats = null) {
            el.components.replaceChildren();
            const rawRows = manifest?.components || manifest?.object?.components || manifest?.bbox?.components || [];
            const rows = Array.isArray(rawRows) ? rawRows.slice() : Object.entries(rawRows).map(([id, value]) => ({ id, ...value }));
            if (manifest?.scene?.components) rows.unshift(...manifest.scene.components.map((v, i) => ({ ...v, id: `Scene ${i + 1} · ${v.encoder}` })));
            if (!rows.length) { el.components.append(h("p", "pruning-help", "Exact encoder/head dimensions, transforms and reuse outcomes are reported by the producer. Missing legacy crop heads are shown rather than silently dropped.")); return; }
            const table = h("table", "pruning-components-table"), header = h("tr");
            for (const title of ["Component / scope", "Model / aggregation", "Weight / dimensions", "Size / input policy", "Availability / reuse"]) header.append(h("th", "", title)); table.append(header);
            for (const produced of rows) {
                const id = produced.id || produced.component;
                const configured = id?.startsWith("scene_") ? recipe?.scene?.components?.[Number(id.slice(6))] : recipe?.bbox?.components?.find(v => `bbox_${v.id}` === id);
                const v = { ...configured, ...produced }, row = h("tr");
                const adjustment = typeof v.adjustment === "object" ? v.adjustment?.mode : v.adjustment;
                const preparation = v.preparation ? `${v.preparation.canonical_size || "native"} · ${v.preparation.embedding_view_mode || "recorded views"}` : v.prepared_input_size || (id?.startsWith("scene_") ? "whole image" : "recorded");
                const input = v.encoder_input ?? v.encoder_input_size;
                const encoderInput = input && typeof input === "object" ? `${Array.isArray(input.size) ? input.size.join("×") : "recorded"}${input.resample ? ` · ${input.resample}` : ""}` : input || "recorded";
                const cache = v.cache || cacheStats?.components?.[id];
                const availability = cache ? `${number(cache.reused || 0)} reused · ${number(cache.encoded || 0)} encoded${cache.producer_cache_hits ? ` · ${number(cache.producer_cache_hits)} producer cache hits` : ""}` : v.status || v.reuse_state || "producer attestation required";
                const raw = v.raw_recipe || {};
                const head = v.head_id || raw.head_id;
                const cells = [v.label || id || v.encoder_type, [v.model || v.encoder_model || raw.model, v.pooling || v.aggregation || raw.pooling, head ? `head ${head}` : ""].filter(Boolean).join(" · "), `${v.weight === undefined ? "resolved weight" : pct(v.weight)} · ${v.dimension || v.dim || v.actual_dimension || "producer-measured"}D`, `${adjustment || v.size_bias || "recorded policy"} · prep ${preparation} → encoder ${encoderInput}`, availability];
                for (const cell of cells) row.append(h("td", "", cell)); table.append(row);
            }
            const wrap = h("div", "pruning-table-wrap"); wrap.append(table); el.components.append(wrap);
            const details = h("details", "pruning-disclosure"); details.append(h("summary", "", "Producer and fitting provenance"), h("pre", "pruning-help", JSON.stringify(manifest, null, 2))); el.components.append(details);
        }
        function sizeDiagnostics(value) {
            el.sizeDiagnostics.replaceChildren();
            if (!value) { el.sizeDiagnostics.append(h("p", "", "The selected policy is recorded per component. Numerical raw/adjusted PCA and covariate checks appear when produced; missing diagnostics are unmeasured.")); return; }
            const rows = Array.isArray(value) ? value : value.components || [value];
            for (const v of rows) el.sizeDiagnostics.append(h("p", "", `${v.component || v.scope || "Feature component"}: ${v.applied === false ? `correction not applied (${v.reason || (v.policy === "none" ? "none selected" : "insufficient covariates")})` : v.policy || "independent correction"}. Fit population ${number(v.fit_count ?? v.population_count)}; covariates ${Array.isArray(v.covariates) ? v.covariates.join(", ") : v.covariates || "recorded in transform"}. Raw size association ${v.raw_size_association ?? "unmeasured"}; adjusted ${v.adjusted_size_association ?? "unmeasured"}.`));
        }
        async function capabilities(context) {
            const suffix = context.analysisJobId ? `?analysis_job_id=${encodeURIComponent(context.analysisJobId)}` : "";
            const data = await request(`/capabilities${suffix}`, {}, context); state.capabilities = data; renderSource();
            const inherited = data.inherited_recipe || adapters.getRecipe?.();
            const sharedDefault = inherited?.origin === "shared_dqe_default_preset";
            el.inheritance.textContent = inherited ? `${inherited.label || inherited.recipe_name || (sharedDefault ? "Shared DQE default representation" : "Current DQE representation")} · ${inherited.note || (sharedDefault ? "the existing DQE default preset supplies the component and preparation policy; compatible cached descriptors are reused" : "all compatible components are inherited; missing or incompatible components need an explicit producer bridge")}` : "No completed DQE representation is attached. Open DQE setup or compute the selected object recipe with the shared feature service; pruning does not use the 2D projection or quality rank as similarity.";
            componentTable(inherited); sizeDiagnostics(data.size_diagnostics);
            if (data.object_recipes) {
                const previous = el.objectRecipe.value;
                el.objectRecipe.replaceChildren(); for (const v of data.object_recipes) el.objectRecipe.add(new Option(v.label || v.id, v.id));
                if (Array.from(el.objectRecipe.options).some(v => v.value === previous)) el.objectRecipe.value = previous;
            }
            const inheritedOption = Array.from(el.objectRecipe.options).find(v => v.value === "inherit_dqe");
            if (inheritedOption) inheritedOption.textContent = sharedDefault ? "Use shared DQE default recipe" : "Use current DQE recipe";
            return data;
        }
        function profileSummary() {
            const data = state.profiles; const attached = data?.dataset?.head_ids || data?.datasets?.[sourceKey(state.context)]?.head_ids || [];
            const preferred = data?.dataset?.preferred_recipes || data?.datasets?.[sourceKey(state.context)]?.preferred_recipes || {};
            const text = `${number(attached.length)} attached SALAD profiles${attached.length ? ` · ${attached.map(id => headLabel(heads().find(v => v.id === id) || { id })).join(", ")}` : ""} · preferred recipes: ${Object.keys(preferred).join(", ") || "none selected"}`;
            document.querySelectorAll("[data-pruning-dataset-profiles-summary]").forEach(n => { n.textContent = text; });
        }
        async function readProfiles(context, identity = sourceKey(context)) {
            const data = await request(`/profiles?source_identity=${encodeURIComponent(identity)}`, {}, context);
            if (identity === sourceKey(context)) { state.profiles = data; profileSummary(); }
            return data;
        }
        function loaded(result) {
            if (!Array.isArray(result.curve) || !result.recipe) throw new Error("The backend returned an incomplete pruning result. Reconnect after the analysis is complete.");
            if (state.result && (state.result.selection_revision !== result.selection_revision || digest(state.result.recipe) !== digest(result.recipe))) { invalidateExport(); invalidatePreview(); resetPlanning(); recipeLab?.reset(); state.savedLab = null; }
            state.result = result; state.removed = clamp(result.budget ?? state.removed); renderSource();
            componentTable(result.component_manifest || result.recipe.component_manifest || result.recipe.object_manifest || (result.recipe.bbox ? result.recipe : state.capabilities?.inherited_recipe), result.recipe, result.cache_stats); sizeDiagnostics(result.size_diagnostics || result.recipe.size_diagnostics);
        }
        async function run() {
            if (state.busy || state.exporting || state.pending) return;
            let operation = ++state.operation; const initial = cleanContext(adapters.context()), draft = config();
            state.busy = state.checkpointing = true; controls(); message("Saving current annotations before analysis …");
            try {
                if (draft.evidence_mode !== "objects_only") {
                    if (!draft.scene_components.length) throw new Error("Enable at least one whole-image feature component.");
                    if (draft.scene_components.some(v => v.encoder === "local_salad" && !v.head_id)) throw new Error("Select a compatible SALAD profile for each enabled SALAD component.");
                    if (draft.scene_components.some(v => !Number.isFinite(v.weight) || v.weight <= 0)) throw new Error("Enabled whole-image components need positive fusion weights. Disable a component to remove it.");
                }
                if (draft.evidence_mode !== "scene_only" && draft.object_components && (!draft.object_components.some(v => v.enabled) || draft.object_components.some(v => v.enabled && (!Number.isFinite(v.weight) || v.weight <= 0)))) throw new Error("Enable at least one object component; enabled components need positive weights.");
                await adapters.checkpoint();
                if (operation !== state.operation) return;
                const fresh = cleanContext(adapters.context());
                if (signature({ ...initial, revision: fresh.revision }) !== signature(fresh)) throw new Error("The dataset changed while saving. Analyze the currently open dataset again.");
                const context = begin(fresh); applyConfig(draft); operation = state.operation; const token = operation; state.busy = true; state.checkpointing = false; controls();
                if (!context.annotationSource?.id) throw new Error("Open or attach the dataset annotation workspace first.");
                recipeLab?.reset(); state.savedLab = null;
                resetPlanning(); state.jobId = ""; state.result = null; state.plan = null; state.review = null; invalidateExport();
                const job = await mutation("analysis", "/jobs", { analysis_job_id: context.analysisJobId || "", annotation_source: context.annotationSource, expected_revision: context.revision, salad_head_id: draft.scene_components.find(v => v.head_id)?.head_id || "auto", config: draft }, context);
                if (!current(context)) return;
                state.jobId = job.job_id; remember(); await poll(context, false);
                if (token === state.operation) { state.busy = false; controls(); }
            } catch (error) { if (operation === state.operation && error.name !== "AbortError") message(error.message, true); }
            finally { if (operation === state.operation) { state.busy = false; state.checkpointing = false; controls(); } }
        }
        async function poll(context, restored = false) {
            const jobId = state.jobId;
            while (current(context) && state.jobId === jobId) {
                const job = await request(`/jobs/${encodeURIComponent(jobId)}`, {}, context);
                message(`${job.message || job.phase || "Analyzing"}${["queued", "running"].includes(job.status) ? ` · ${Math.round((job.progress || 0) * 100)}%` : ""}`);
                if (job.status === "completed") {
                    loaded(await request(`/jobs/${encodeURIComponent(jobId)}/result`, {}, context));
                    state.removed = clamp(restored ? state.removed : Math.floor(state.result.image_count * .2));
                    state.review = await request(`/jobs/${encodeURIComponent(jobId)}/review`, {}, context);
                    state.lastBatch = lastBatch(state.review); syncPlanning(true); render(); await families(context, true); if (state.planning.view === "impact") await impactCards(context, true); return;
                }
                if (!["queued", "running"].includes(job.status)) throw new Error(job.message || `Analysis ${job.status}`);
                await pause(1000);
            }
        }
        function render() {
            const r = state.result; if (!r) { controls(); return; }
            el.slider.max = el.count.max = r.image_count; el.percent.max = 100;
            if (!state.planning.active) syncPlanning();
            const ceiling = state.planning.draft.max_removed_images;
            el.slider.setAttribute("aria-valuetext", `Propose at most ${ceiling} images (${(100 * ceiling / r.image_count).toFixed(2)}%)`);
            const plan = displayPlan(); const proposed = plan?.removed_count ?? state.plan?.removed_count ?? state.removed; const approved = appliedPlan().approved_count ?? state.review?.approved_count ?? r.approved_count ?? 0;
            const previewReady = state.planning.preview?.status === "ready";
            el.summary.textContent = planningDirty() && !previewReady ? `Preview pending for a ceiling of ${number(ceiling)} images. The applied plan currently proposes ${number(proposed)} removals; ${number(approved)} are approved.` : `${planningDirty() ? "Preview" : "Applied"} ceiling ${number(ceiling)} images (${(100 * ceiling / r.image_count).toFixed(2)}%) · ${number(proposed)} proposed · ${number(approved)} approved in applied plan · ${number(r.image_count - proposed)} retained in ${planningDirty() ? "preview" : "plan"}`;
            el.exportPreview.textContent = `The reviewed variant removes ${number(approved)} approved images and retains ${number(r.image_count - approved)} of ${number(r.image_count)} images. ${number(Math.max(0, proposed - approved))} unreviewed proposals remain in the export.`;
            const s = r.cache_stats || {};
            const missingCaches = (r.dependency_status?.optional_caches || []).filter(value => value.status !== "available").length;
            const descriptorCounts = [["Reused", s.scene_reused, s.bbox_reused], ["Encoded", s.scene_encoded, s.bbox_encoded]].map(([verb, scene, objects]) => {
                const counts = [scene == null ? "" : `${number(scene)} whole-image`, objects == null ? "" : `${number(objects)} object`].filter(Boolean);
                return counts.length ? `${verb} ${counts.join(" and ")} descriptors` : "";
            }).filter(Boolean).join("; ");
            el.cache.textContent = `${r.recipe.scene_note || "Feature producer reuse is recorded per branch."} ${descriptorCounts ? `${descriptorCounts}.` : "Descriptor counts were not recorded for this saved analysis."} ${s.derived_reused ? "Cached matching reused." : s.assignments_reused ? "Exact cached assignments reused with the current gates and index weights." : "Matching is recipe-bound."}${missingCaches ? " Optional feature caches are missing or were cleared; the saved review remains available." : ""}`;
            const activeConfig = r.resolved_config || r.config || r.recipe.config || {};
            const activeDemoBound = (r.recipe.qualification || r.qualification || "").startsWith("exact WALDO40 raw demo feature binding");
            const activeBudgeted = state.planning.active?.policy === "budgeted_coverage";
            el.activeRecipe.textContent = activeBudgeted ? `Saved features · source revision ${r.source_revision ?? state.context.revision}. Removal needs a direct retained scene match ≥ ${pct(activeConfig.scene_threshold ?? .92)} (${pct(activeConfig.negative_scene_threshold ?? .94)} between explicit negatives). Objects can match different retained images; uncovered appearances must fit both loss ceilings.` : `Active analysis ${state.jobId} · source revision ${r.source_revision ?? state.context.revision} · ${activeConfig.evidence_mode || "compound"} · ${activeConfig.object_recipe || r.recipe.object?.name || "resolved object recipe"} · ${activeConfig.object_adjustment || "recorded adjustment policy"} · thresholds ${activeDemoBound ? "raw-demo starting settings" : "unqualified for this feature space"}. Setup changes above do not change these results until analyzed.`;
            const requiredScene = (activeConfig.evidence_mode || "compound") !== "objects_only" && (activeConfig.scene_requirement || "required") === "required";
            if (!activeBudgeted) el.activeRecipe.append(document.createTextNode(` ${requiredScene ? `Whole-image scene similarity must reach ${pct(activeConfig.scene_threshold ?? .92)} even when objects are very close; retrieval skips pairs below ${pct(r.similarity_recipe?.pair_scene_prefilter ?? Math.min(.65, activeConfig.scene_threshold ?? .92, activeConfig.negative_scene_threshold ?? .94))}. Choose optional scene coverage to explore matching objects against different backgrounds.` : "Whole-image scene similarity is not a removal gate; object coverage remains independently checked."}`));
            if (r.retrieval?.length) el.cache.append(document.createTextNode(` ${activeBudgeted ? "Original analysis discovery: " : ""}${r.retrieval.map(v => `${v.route || v.component || "Object route"}: ${v.version || v.method || "recorded retrieval"}, ${number(v.query_count ?? v.rows)} queries, ${number(v.zero_cross_image_queries ?? 0)} with no cross-image candidates`).join(". ")}. ${activeBudgeted ? "Coverage planning resolves direct matches from the frozen cached features separately." : "Retrieval recall is separate from the final redundancy score."}`));
            el.keep.replaceChildren(); const keepIds = state.review?.keep_ids || (state.review?.pins ? Object.entries(state.review.pins).filter(([, field]) => field.value).map(([id]) => id) : r.keep_ids) || [];
            for (const value of Array.isArray(keepIds) ? keepIds : Object.keys(keepIds)) {
                const id = typeof value === "string" ? value : value.image_id; const b = button(`Keep ${imageLabel(id)} · Undo pin`, () => reviewAction({ action: "pin", image_id: id, keep: false }), "pruning-pin"); b.dataset.reviewAction = ""; el.keep.append(b);
            }
            renderPlanning(); chart(); controls(); remember();
            document.querySelectorAll("[data-pruning-dqe-summary]").forEach(n => { n.textContent = ` ${number(proposed)} proposals · ${number(approved)} approved.`; });
        }
        function lastBatch(snapshot) {
            if (snapshot?.last_batch_id) return snapshot.last_batch_id;
            const values = Object.values(snapshot?.batches || {}).filter(v => !v.undone && ["bulk_commit", "apply_plan", "pin", "confirm_plan"].includes(v.action)).sort((a, b) => (a.at || 0) - (b.at || 0));
            return values.at(-1)?.batch_id || "";
        }
        function imageLabel(id) {
            for (const family of state.families.values()) { const image = (family.members || []).find(v => idOf(v) === id); if (image) { if (image.display_id !== undefined) return `#${image.display_id}`; if (image.image_index !== undefined) return `#${image.image_index + 1}`; const name = String(image.image_relpath || id).split("/").pop(); return name.length > 24 ? `${name.slice(0, 20)}…` : name; } }
            return String(id).slice(0, 12);
        }
        function chartView() {
            const ready = state.planning.preview?.status === "ready";
            const plan = ready ? displayPlan() : { ...state.result, ...appliedPlan(), ...state.review };
            const settings = normalizeSettings(ready ? plan.settings || plan.active_settings : state.planning.active || plan.active_settings);
            const curve = Array.isArray(plan.curve) ? plan.curve : [];
            return { plan, settings, curve, outdated: !ready && !!plan.accounting_outdated, source: ready ? "preview" : "applied", label: ready ? "Preview" : "Applied plan",
                note: planningDirty() && !ready ? "Applied plan shown until a new preview is ready. " : `${ready ? "Preview" : "Applied plan"}. ` };
        }
        const chartExtent = (view = chartView()) => 100 * view.curve.reduce((max, point) => Math.max(max, Number(point.removed_count) || 0), Math.max(1, view.settings.max_removed_images)) / state.result.image_count;
        function lossChart(view) {
            const { curve, settings } = view;
            if (settings.policy !== "budgeted_coverage") return false;
            const svg = el.chart, r = state.result, ceiling = settings.max_removed_images;
            svg.replaceChildren();
            const draw = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const [key, value] of Object.entries(attrs)) n.setAttribute(key, value); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
            const extent = curve.reduce((max, point) => Math.max(max, Number(point.removed_count) || 0), Math.max(1, ceiling));
            const top = Math.min(1, curve.reduce((max, point) => Math.max(max, Number(point.lost_appearance_weight_fraction) || 0), Math.max(.01, settings.max_lost_appearance_weight_fraction)) * 1.08);
            svg.dataset.maximumImages = extent;
            const x = value => 58 + value / extent * 580, y = value => 230 - value / top * 190;
            for (let i = 0; i <= 4; i++) { const v = top * i / 4; draw("line", { x1: 58, x2: 638, y1: y(v), y2: y(v), stroke: "#d9dee5" }); draw("text", { x: 50, y: y(v) + 4, "text-anchor": "end", class: "pruning-chart-label" }, pct(v, 1)); }
            for (let i = 0; i <= 5; i++) { const v = extent * i / 5; draw("text", { x: x(v), y: 250, "text-anchor": "middle", class: "pruning-chart-label" }, `${(100 * v / r.image_count).toFixed(1)}%`); }
            draw("text", { x: 58, y: 20, class: "pruning-chart-label" }, `${view.label} · rarity-weighted appearance loss`);
            draw("text", { x: 348, y: 276, "text-anchor": "middle", class: "pruning-chart-label" }, "Images proposed for removal · percentage of source");
            const stride = Math.max(1, Math.ceil(curve.length / 500));
            const rows = curve.filter((v, i) => finiteNumber(v.lost_appearance_weight_fraction) && (i % stride === 0 || i === curve.length - 1));
            draw("polyline", { points: rows.map(v => `${x(v.removed_count)},${y(v.lost_appearance_weight_fraction)}`).join(" "), fill: "none", stroke: "#3269d5", "stroke-width": 3 });
            draw("line", { x1: 58, x2: 638, y1: y(settings.max_lost_appearance_weight_fraction), y2: y(settings.max_lost_appearance_weight_fraction), stroke: "#c17b23", "stroke-dasharray": "5 4" });
            draw("line", { x1: x(ceiling), x2: x(ceiling), y1: 32, y2: 230, stroke: "#52677b", "stroke-dasharray": "4 3" });
            el.coverage.textContent = `${view.note}Blue: measured weighted loss for verified plan prefixes. Dashed gold: ${pct(settings.max_lost_appearance_weight_fraction)} weighted-loss ceiling. The separate cap of ${number(settings.max_affected_appearance_groups)} affected groups also applies. Ranking: ${pct(settings.object_weight)} objects + ${pct(1 - settings.object_weight)} scenes. Keeping an image can restore part of a group's coverage without restoring the entire group.`;
            return true;
        }
        function chart() {
            const view = chartView(), { plan, settings, curve } = view;
            el.chart.dataset.planSource = view.source; el.chart.dataset.policy = settings.policy;
            if (view.outdated) {
                el.chart.replaceChildren(); el.chart.dataset.planSource = "outdated"; el.chart.dataset.maximumImages = 0;
                const label = document.createElementNS(NS, "text");
                for (const [key, value] of Object.entries({ x: 348, y: 140, "text-anchor": "middle", class: "pruning-chart-label" })) label.setAttribute(key, value);
                label.textContent = "Recalculate proposals to update coverage"; el.chart.append(label);
                el.coverage.textContent = "Previous coverage is unverified under the current rules. Its graph is hidden until a new preview is ready.";
                return;
            }
            if (lossChart(view)) return;
            const svg = el.chart, r = state.result; svg.replaceChildren();
            const node = (tag, attrs, value) => { const n = document.createElementNS(NS, tag); for (const [key, v] of Object.entries(attrs)) n.setAttribute(key, v); if (value !== undefined) n.textContent = value; svg.append(n); return n; };
            const fields = [["coverage", "#3269d5"], ["scene_coverage", "#299276"], ["object_coverage", "#9560b6"], ["worst_class_coverage", "#c17b23"]]; let minimum = 1;
            for (const p of curve) for (const [f] of fields) if (Number.isFinite(p[f])) minimum = Math.min(minimum, p[f]);
            const lower = Math.max(0, 1 - Math.max(.0001, 1 - minimum) * 1.1), extent = chartExtent(view);
            svg.dataset.maximumImages = extent * r.image_count / 100;
            const x = v => 58 + v / extent * 580, y = v => 230 - (v - lower) / (1 - lower) * 190;
            for (let tick = 0; tick <= 4; tick++) { const value = lower + tick / 4 * (1 - lower); node("line", { x1: 58, y1: y(value), x2: 638, y2: y(value), stroke: "#d9dee5" }); node("text", { x: 50, y: y(value) + 4, "text-anchor": "end", class: "pruning-chart-label" }, pct(value, 3)); }
            for (let tick = 0; tick <= 5; tick++) { const value = extent * tick / 5; node("text", { x: x(value), y: 250, "text-anchor": "middle", class: "pruning-chart-label" }, `${value.toFixed(2)}%`); }
            node("text", { x: 348, y: 276, "text-anchor": "middle", class: "pruning-chart-label" }, "Maximum images proposed for removal · complete dataset"); node("text", { x: 58, y: 20, class: "pruning-chart-label" }, `${view.label} · estimated retained feature coverage`);
            const step = Math.max(1, Math.floor(curve.length / 500));
            for (const [f, color] of fields) { const points = curve.filter((p, i) => Number.isFinite(p[f]) && (i % step === 0 || i === curve.length - 1)).map(p => `${x(100 * p.removed_count / r.image_count)},${y(p[f])}`).join(" "); node("polyline", { points, fill: "none", stroke: color, "stroke-width": f === "coverage" ? 3 : 1.5 }); }
            const maximum = curve.reduce((max, point) => Math.max(max, Number(point.removed_count) || 0), 0);
            const proposed = plan.removed_count ?? Math.min(settings.max_removed_images, maximum);
            const p = curve.find(point => point.removed_count === proposed);
            const coverage = plan.coverage || plan.metrics || plan.chosen || p || {};
            node("line", { x1: x(100 * settings.max_removed_images / r.image_count), x2: x(100 * settings.max_removed_images / r.image_count), y1: 32, y2: 230, stroke: "#27334d", "stroke-dasharray": "4 3" });
            if (Number.isFinite(coverage.coverage)) node("circle", { cx: x(100 * proposed / r.image_count), cy: y(coverage.coverage), r: 5, fill: "#3269d5" });
            el.coverage.textContent = `${view.note}Combined ${pct(coverage.coverage, 3)} · scene ${pct(coverage.scene_coverage, 3)} · objects ${pct(coverage.object_coverage, 3)} · worst class ${pct(coverage.worst_class_coverage, 3)}. Index: ${pct(settings.object_weight)} objects + ${pct(1 - settings.object_weight)} scenes. Maximum eligible: ${number(maximum)} images (${(100 * maximum / r.image_count).toFixed(2)}%).`;
        }
        function budget(value) {
            editSettings({ max_removed_images: clamp(value) });
        }
        function handleError(error, context = state.context) {
            if (error.name === "AbortError" || !current(context)) return;
            message(error.message, true); controls();
        }
        const ids = value => new Set(Array.isArray(value) ? value.map(v => typeof v === "string" ? v : v.image_id) : Object.entries(value || {}).filter(([, field]) => field === true || field?.value === true).map(([id]) => id));
        function thumbnail(image, context, caption, compact = false) {
            const figure = h("figure"), frame = h("div", "pruning-image"), img = h("img");
            img.alt = `${caption}: ${image.image_relpath || idOf(image)}`; img.loading = "lazy"; img.decoding = "async";
            img.src = `${context.apiRoot}/dataset_pruning/jobs/${encodeURIComponent(state.jobId)}/images/${encodeURIComponent(idOf(image))}`;
            frame.style.aspectRatio = `${image.width || 1} / ${image.height || 1}`;
            const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 1000 1000"); svg.setAttribute("preserveAspectRatio", "none"); svg.setAttribute("aria-hidden", "true");
            for (const object of image.objects || []) {
                if (!Array.isArray(object.box)) continue;
                const [x, y, w, height] = object.box, rect = document.createElementNS(NS, "rect");
                for (const [k, v] of Object.entries({ x: (x - w / 2) * 1000, y: (y - height / 2) * 1000, width: w * 1000, height: height * 1000, fill: "none", stroke: "#65d8ff", "stroke-width": compact ? 5 : 3 })) rect.setAttribute(k, v);
                const title = document.createElementNS(NS, "title"); title.textContent = `${className(object.class_id)} · label row ${object.label_row ?? object.row ?? "recorded"}`; rect.append(title); svg.append(rect);
            }
            const label = image.display_id !== undefined ? `#${image.display_id}` : image.image_index !== undefined ? `#${image.image_index + 1}` : imageLabel(idOf(image));
            const detail = h("figcaption", "", `${caption} · ${label} · ${(image.objects || []).length} objects`);
            detail.title = `${image.image_relpath || idOf(image)} · ${idOf(image)}`;
            frame.append(img, svg); figure.append(frame, detail); return figure;
        }
        function renderImpactCards(context) {
            const p = state.planning; el.impact.replaceChildren();
            for (const row of p.rows) {
                const candidate = row.candidate, witness = row.scene_witness;
                if (!candidate || !idOf(candidate)) continue;
                const id = idOf(candidate), witnessId = witness ? idOf(witness) : row.witness_id;
                const card = h("article", "pruning-impact-card"); card.dataset.candidateId = id;
                const label = candidate.display_id !== undefined ? `#${candidate.display_id}` : candidate.image_index !== undefined ? `#${candidate.image_index + 1}` : String(candidate.image_relpath || id).split("/").pop();
                const restorationMeasured = finiteNumber(row.restored_appearance_groups) || finiteNumber(row.restored_objects);
                const heading = h("div", "pruning-section-heading"); heading.append(h("h4", "", label), h("span", "pruning-badge", restorationMeasured && finiteNumber(row.restoration_rank) ? `Restoration rank ${number(row.restoration_rank)}` : "Strict duplicate review")); card.append(heading);
                const frames = h("div", "pruning-impact-frames");
                frames.append(thumbnail(candidate, context, "Proposed removal", true));
                if (witness && idOf(witness)) frames.append(thumbnail(witness, context, "Retained scene witness", true));
                card.append(frames);
                const restoration = h("p", "pruning-impact-restoration");
                restoration.textContent = `Keeping this image: ${measuredCount(row.restored_appearance_groups)} groups regain full coverage · ${measuredPercent(row.restored_appearance_weight)} rarity-weighted coverage restored${finiteNumber(row.restored_objects) ? ` · ${number(row.restored_objects)} original objects regain a match` : ""}.`;
                card.append(restoration);
                if (finiteNumber(row.partially_restored_groups) && row.partially_restored_groups > 0) card.append(h("p", "pruning-help", `${number(row.partially_restored_groups)} more groups regain some object coverage. They still count as affected until every original member is covered.`));
                const missing = row.uncovered_objects || [];
                const details = h("details", "pruning-disclosure"); details.open = missing.length > 0 && missing.length <= 3; details.append(h("summary", "", `${number(missing.length)} uncovered objects in this image · native cutouts`));
                const crops = h("div", "pruning-impact-crops");
                for (const object of missing) {
                    const figure = h("figure");
                    if (object.crop_url) { const image = h("img"); image.src = assetURL(object.crop_url, context); image.alt = `${className(object.class_id)} · original object row ${object.row}`; image.loading = "lazy"; image.decoding = "async"; figure.append(image); }
                    figure.append(h("figcaption", "", `${className(object.class_id)} · row ${object.row}${object.width && object.height ? ` · ${object.width} × ${object.height} original pixels` : ""}`)); crops.append(figure);
                }
                if (!missing.length) crops.append(h("p", "pruning-help", "No uncovered object cutouts are recorded for this candidate. Its removal can still change scene coverage."));
                details.append(crops); card.append(details);
                if (row.object_witnesses?.length) {
                    const refs = h("details", "pruning-disclosure"); refs.append(h("summary", "", "Objects can have different retained witnesses"));
                    for (const object of row.object_witnesses) { const target = object.witness_image_id || object.witness_id || object.image_id; refs.append(h("p", "pruning-help", `${className(object.class_id)} · row ${object.row ?? object.label_row ?? object.candidate_row} → ${target ? imageLabel(target) : "no retained object witness"} · ${measuredPercent(object.appearance ?? object.similarity)}`)); }
                    card.append(refs);
                }
                const approved = row.status === "approved";
                card.append(h("p", `pruning-badge${approved ? " good" : ""}`, approved ? "Approved in the applied plan" : planningDirty() ? "Preview only · apply before reviewing" : "Unreviewed · retained in export"));
                const actions = h("div", "pruning-inline");
                const keep = button("Keep this image", () => reviewAction({ action: "pin", image_id: id, keep: true }), "pruning-keep"); keep.dataset.reviewAction = "";
                const approve = button(approved ? "Undo approval" : "Approve this removal", () => reviewAction(approved ? { action: "undo_decision", candidate_id: id } : { action: "approve", candidate_id: id, witness_id: witnessId, evidence_digest: row.evidence_digest }), "pruning-approve"); approve.dataset.reviewAction = ""; approve.dataset.requiresConfirmation = String(!approved && (row.requires_confirmation === true || !witnessId || !row.evidence_digest));
                actions.append(keep, approve);
                if (row.family_id) actions.append(button("Open family", async () => { if (await setReviewView("family")) await locateFamily(row.family_id); }));
                if (state.planning.draft?.policy === "budgeted_coverage") actions.append(button("Inspect object coverage", () => objectCoverageDialog(row)));
                else if (witnessId) actions.append(button("Inspect exact comparison", () => comparison(id, witnessId)));
                card.append(actions);
                if (row.requires_confirmation) card.append(h("p", "pruning-help", "Confirm this strict family's keepers in Families before individual approval."));
                el.impact.append(card);
            }
            if (!p.rows.length) el.impact.append(h("p", "pruning-help", p.pending ? "Waiting for the current preview …" : "No proposed removals in this plan. Raise the image ceiling or adjust the explicit loss limits, then inspect the preview."));
            renderPlanning(); controls();
        }
        async function impactCards(context, reset = false) {
            if (!state.result || !current(context)) return;
            const p = state.planning;
            if (p.dirty && p.preview?.status !== "ready") return;
            if (!reset && (p.loading || p.next === null)) return;
            const ticket = ++p.listSequence, jobId = state.jobId, previewId = p.dirty ? p.preview.preview_id : null, binding = reviewPayload(), offset = reset ? 0 : p.next;
            p.loading = true; renderPlanning(); controls();
            try {
                const data = await request(`/jobs/${encodeURIComponent(jobId)}/selection`, post({ ...binding, ...(previewId ? { preview_id: previewId } : {}), view: "impact", offset, limit: 18 }), context);
                if (!current(context) || state.jobId !== jobId || ticket !== p.listSequence || previewId !== (p.dirty ? p.preview?.preview_id : null) || binding.expected_review_revision !== reviewPayload().expected_review_revision || binding.expected_plan_revision !== reviewPayload().expected_plan_revision) return;
                const rows = data.rows || data.cards || [];
                if (reset) p.rows = rows; else { const known = new Set(p.rows.map(row => idOf(row.candidate || {}))); p.rows.push(...rows.filter(row => !known.has(idOf(row.candidate || {})))); }
                p.next = data.next_offset ?? null;
                if (previewId) p.preview.plan = { ...p.preview.plan, ...(data.plan || {}), ...Object.fromEntries(["proposed_loss", "approved_loss", "appearance_group_count", "total_appearance_weight", "curve"].filter(key => data[key] !== undefined).map(key => [key, data[key]])) };
                else { state.plan = { ...state.plan, ...(data.plan || {}), ...Object.fromEntries(["proposed_loss", "approved_loss", "appearance_group_count", "total_appearance_weight", "curve"].filter(key => data[key] !== undefined).map(key => [key, data[key]])) }; }
                renderImpactCards(context);
            } catch (error) { if (ticket === p.listSequence && current(context)) { p.error = error.message; el.impact.append(h("p", "pruning-error", error.message)); } }
            finally { if (ticket === p.listSequence) { p.loading = false; renderPlanning(); controls(); } }
        }
        async function setReviewView(view) {
            if (state.busy || state.exporting || state.externalRefreshing || state.pending || state.recoveryRequired) return false;
            state.planning.view = view; renderPlanning(); remember();
            if (view === "impact") await impactCards(state.context, true);
            else await families(state.context, true);
            return true;
        }
        function familyGraph(family) {
            const svg = document.createElementNS(NS, "svg"); svg.classList.add("pruning-family-graph"); svg.setAttribute("viewBox", "0 0 340 230"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Directed verified object coverage between family images. A line does not imply all members are interchangeable.");
            const members = family.members || [], map = new Map(), points = family.graph?.points || [], positioned = points.length && points.every(v => Number.isFinite(v.x) && Number.isFinite(v.y)) && (Math.max(...points.map(v => v.x)) - Math.min(...points.map(v => v.x)) + Math.max(...points.map(v => v.y)) - Math.min(...points.map(v => v.y)) > 1e-12);
            if (positioned) { const xs = points.map(v => v.x), ys = points.map(v => v.y), xmin = Math.min(...xs), ymin = Math.min(...ys), dx = Math.max(.001, Math.max(...xs) - xmin), dy = Math.max(.001, Math.max(...ys) - ymin); for (const v of points) map.set(v.image_id, [30 + (v.x - xmin) / dx * 280, 28 + (v.y - ymin) / dy * 170]); }
            for (let i = 0; i < members.length; i++) if (!map.has(idOf(members[i]))) { const a = i / Math.max(1, members.length) * Math.PI * 2; map.set(idOf(members[i]), [170 + Math.cos(a) * 120, 110 + Math.sin(a) * 75]); }
            const keepers = ids(family.keeper_ids), pins = ids(family.pins), hard = ids(family.hard_keep_ids);
            const draw = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
            const markerId = `pruningArrow${family.id}`, defs = document.createElementNS(NS, "defs"), marker = document.createElementNS(NS, "marker"), arrow = document.createElementNS(NS, "path"); for (const [k, v] of Object.entries({ id: markerId, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" })) marker.setAttribute(k, v); arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z"); arrow.setAttribute("fill", "#779baa"); marker.append(arrow); defs.append(marker); svg.append(defs);
            for (const e of family.graph?.edges || []) { const left = map.get(e.candidate_id), right = map.get(e.witness_id); if (!left || !right) continue; const distance = Math.max(.001, Math.hypot(right[0] - left[0], right[1] - left[1])), dx = (right[0] - left[0]) / distance, dy = (right[1] - left[1]) / distance; const line = draw("line", { x1: left[0] + dx * 8, y1: left[1] + dy * 8, x2: right[0] - dx * 10, y2: right[1] - dy * 10, "marker-end": `url(#${markerId})`, stroke: keepers.has(e.witness_id) ? "#77b89f" : "#d2dce5", "stroke-width": keepers.has(e.witness_id) ? 2 : 1 }); const title = document.createElementNS(NS, "title"); title.textContent = `${imageLabel(e.candidate_id)} → ${imageLabel(e.witness_id)} · ${pct(e.combined)} · weakest object ${pct(e.minimum_object_appearance)}`; line.append(title); }
            for (const image of members) { const id = idOf(image), point = map.get(id); const n = draw("circle", { cx: point[0], cy: point[1], r: keepers.has(id) ? 8 : 5, fill: keepers.has(id) ? "#28835c" : "#8094ab", stroke: pins.has(id) || hard.has(id) ? "#c58c21" : "#fff", "stroke-width": pins.has(id) || hard.has(id) ? 3 : 1, tabindex: 0 }); n.setAttribute("role", "button"); n.setAttribute("aria-label", `Inspect ${image.image_relpath || id}`); const open = () => memberDetails(family, image); n.addEventListener("click", open); n.addEventListener("keydown", event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); open(); } }); }
            draw("text", { x: 170, y: 220, "text-anchor": "middle", class: "pruning-chart-label" }, positioned ? "Feature layout · direct coverage edges" : "Coverage graph · layout is not feature distance"); return svg;
        }
        function familyCurve(family) {
            const values = family.coverage_curve || family.graph?.curve || [];
            if (!values.length) return h("p", "pruning-help", "The server computes the family keeper sequence and direct coverage. A local descriptor-coverage curve is not yet available for this family.");
            const svg = document.createElementNS(NS, "svg"); svg.classList.add("pruning-family-curve"); svg.setAttribute("viewBox", "0 0 340 180"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Retained representatives versus estimated family feature coverage. Direct witness gates remain separate.");
            const draw = (tag, attrs, value) => { const n = document.createElementNS(NS, tag); for (const [key, v] of Object.entries(attrs)) n.setAttribute(key, v); if (value !== undefined) n.textContent = value; svg.append(n); return n; };
            const count = v => v.keeper_count ?? v.keep_count ?? v.retained_count;
            const fields = [["coverage", "#3269d5"], ["object_coverage", "#9560b6"], ["scene_coverage", "#299276"]];
            const measurable = values.flatMap(v => fields.map(([key]) => v[key]).filter(Number.isFinite));
            const minimum = measurable.length ? Math.max(0, Math.min(...measurable) - .005) : 0, maximum = Math.max(1, ...values.map(v => Number(count(v)) || 1));
            const x = v => 42 + (v - 1) / Math.max(1, maximum - 1) * 278, y = v => 140 - (v - minimum) / Math.max(.0001, 1 - minimum) * 105;
            for (let i = 0; i <= 2; i++) { const value = minimum + (1 - minimum) * i / 2; draw("line", { x1: 42, x2: 320, y1: y(value), y2: y(value), stroke: "#d7e2ea" }); draw("text", { x: 38, y: y(value) + 4, "text-anchor": "end", class: "pruning-chart-label" }, pct(value, 1)); }
            for (const [key, color] of fields) { const points = values.filter(v => Number.isFinite(v[key]) && Number.isFinite(Number(count(v)))).map(v => `${x(count(v))},${y(v[key])}`).join(" "); draw("polyline", { points, fill: "none", stroke: color, "stroke-width": 2 }); }
            const currentCount = family.keeper_ids?.length || 1; draw("line", { x1: x(currentCount), x2: x(currentCount), y1: 35, y2: 140, stroke: "#28835c", "stroke-dasharray": "3 3" });
            draw("text", { x: 42, y: 158, class: "pruning-chart-label" }, "1"); draw("text", { x: 320, y: 158, "text-anchor": "end", class: "pruning-chart-label" }, String(maximum)); draw("text", { x: 180, y: 177, "text-anchor": "middle", class: "pruning-chart-label" }, "Images retained in family · coverage estimate"); return svg;
        }
        function renderFamily(family, context) {
            const budgeted = state.planning.draft?.policy === "budgeted_coverage";
            const familyId = family.id, old = state.familyNodes.get(familyId); const detailsOpen = Array.from(old?.querySelectorAll("details") || [], n => n.open);
            const card = h("article", "pruning-family"); card.dataset.familyId = familyId;
            const total = family.total_member_count ?? family.members?.length ?? 0;
            const heading = h("div", "pruning-family-heading"), title = h("h4", "", `Family ${family.label || familyId.slice(0, 10)} · ${number(total)} images`);
            const badge = h("span", `pruning-badge ${family.confirmed ? "good" : "warning"}`, family.confirmed ? "Keepers confirmed" : "Suggested keepers"); heading.append(title, badge);
            const body = h("div", "pruning-family-body"), summary = h("p", "pruning-family-summary", `${number(family.keeper_ids?.length)} retained representatives · ${number(family.proposals?.length)} proposed removals · ${number((family.proposals || []).filter(v => v.status === "approved").length)} approved. ${budgeted ? "Each removal has a direct retained scene witness. Object witnesses may be in other families; permitted object losses stay within both budgets." : "Each removal has a direct surviving witness."}`);
            const draft = state.familyDrafts.get(familyId) || { count: family.requested_count || family.keeper_ids?.length || 1, keeperIds: null, preview: null, serial: 0 }; state.familyDrafts.set(familyId, draft);
            const controlsRow = h("div", "pruning-family-controls"), slider = old?.querySelector('[data-family-count]') || input("range", draft.count, { min: "1", max: String(total || 1), step: "1" });
            slider.dataset.familyCount = familyId; slider.max = total || 1; slider.value = draft.count; const output = h("output", "", `${draft.count} of ${total}`);
            if (!old) slider.addEventListener("input", () => { const currentDraft = state.familyDrafts.get(familyId); currentDraft.count = Number(slider.value); currentDraft.keeperIds = null; currentDraft.preview = null; remember(); slider.parentElement.querySelector("output").textContent = `${slider.value} of ${state.families.get(familyId)?.total_member_count ?? state.families.get(familyId)?.members.length ?? 0}`; familyPreview(familyId, true); });
            const label = field("Keep", slider); label.append(output); controlsRow.append(label);
            const previewButton = button("Preview selected keepers", () => familyPreview(familyId)); previewButton.dataset.reviewAction = "";
            const confirmButton = button("Confirm these keepers", () => confirmFamily(familyId), "pruning-primary"); confirmButton.dataset.familyConfirm = familyId; confirmButton.dataset.reviewAction = ""; confirmButton.dataset.requiresConfirmation = String(!draft.preview);
            controlsRow.append(previewButton, confirmButton);
            const confirmation = state.review?.constraints?.[familyId], confirmationBatch = state.review?.batches?.[confirmation?.owner];
            if (family.confirmed && confirmationBatch && !confirmationBatch.undone && confirmationBatch.action === "confirm_plan") { const undoConfirmation = button("Undo keeper confirmation", () => reviewAction({ action: "bulk_undo", batch_id: confirmationBatch.batch_id })); undoConfirmation.dataset.reviewAction = ""; controlsRow.append(undoConfirmation); }
            const previewText = h("p", "pruning-help", draft.preview ? `${number(draft.preview.captured_eligible_count ?? draft.preview.removed_count)} eligible removals in this family · ${number(draft.preview.additional_keeper_count)} additional keepers required` : "Explore a count or select representatives, then preview the exact keeper plan. Pins, measured coverage gaps and unmeasured comparisons may require additional images."); previewText.dataset.familyPreviewStatus = familyId;
            body.append(summary, controlsRow, previewText);
            const direct = family.direct_comparisons;
            if (direct) {
                const missing = h("p", "pruning-help", `${number(direct.measured_count)} direct comparisons to current keepers measured · ${number(direct.measured_failing_or_rejected_count)} fail or were rejected · ${number(direct.unmeasured_count)} unmeasured. Missing comparisons do not establish that an image is novel.`);
                const measure = button("Compute missing comparisons to selected keepers", () => { const currentDraft = state.familyDrafts.get(familyId); const keeperIds = currentDraft.keeperIds || state.families.get(familyId).keeper_ids; return reviewAction({ action: "measure_direct", family_id: familyId, keeper_ids: keeperIds, plan_revision: state.plan.plan_revision }); }); measure.dataset.reviewAction = "";
                body.append(missing, measure, h("p", "pruning-help", "Reuses cached descriptors with this analysis's exact fitted size transforms. No encoder or training runs. If optional descriptors were cleared, recompute the selected analysis first. Preview and confirm keepers after new evidence is saved."));
            }
            if (family.uncovered_ids?.length || family.exception_reasons?.length) body.append(h("p", "pruning-gap", `${family.uncovered_ids?.length ? `${number(family.uncovered_ids.length)} uncovered images stay. ` : ""}${(family.exception_reasons || []).map(v => typeof v === "string" ? v : v.reason || JSON.stringify(v)).join(" ")}`));
            const layout = h("div", "pruning-family-layout"), graph = h("div"); graph.append(familyGraph(family), familyCurve(family), h("p", "pruning-help", "Green: selected keepers. Gold outline: pinned/protected. Direct edges are evidence; connected chains can still need several representatives."));
            const memberGrid = h("div", "pruning-members"), selected = draft.keeperIds ? new Set(draft.keeperIds) : ids(family.keeper_ids), pinned = ids(family.pins), hard = ids(family.hard_keep_ids);
            for (const image of family.members || []) {
                const id = idOf(image), member = h("div", `pruning-member${selected.has(id) ? " is-keeper" : ""}${pinned.has(id) ? " is-pinned" : ""}`); member.append(thumbnail(image, context, selected.has(id) ? "Keeper" : "Member", true));
                const choice = input("checkbox", "", { checked: selected.has(id), disabled: hard.has(id) }); choice.dataset.reviewAction = "";
                choice.addEventListener("change", () => { const currentDraft = state.familyDrafts.get(familyId); const keepers = new Set(currentDraft.keeperIds || state.families.get(familyId).keeper_ids); choice.checked ? keepers.add(id) : keepers.delete(id); currentDraft.keeperIds = Array.from(keepers); currentDraft.count = Math.max(1, keepers.size); currentDraft.preview = null; remember(); member.classList.toggle("is-keeper", choice.checked); familyPreview(familyId, true); }); member.append(field(hard.has(id) ? "Protected keeper" : "Select as keeper", choice));
                const primary = button("Use as primary keeper", () => { const currentDraft = state.familyDrafts.get(familyId); currentDraft.keeperIds = Array.from(new Set([id, ...hard, ...pinned])); currentDraft.count = currentDraft.keeperIds.length; currentDraft.preview = null; remember(); renderFamily(state.families.get(familyId), context); familyPreview(familyId, true); }); primary.dataset.reviewAction = "";
                const pin = button(pinned.has(id) ? "Undo independent Keep" : "Always keep this image", () => reviewAction({ action: "pin", image_id: id, keep: !pinned.has(id) }), "pruning-keep"); pin.dataset.reviewAction = ""; member.append(primary, pin, button("Inspect image and scale", () => memberDetails(family, image))); memberGrid.append(member);
            }
            layout.append(graph, memberGrid); body.append(layout);
            const pairs = h("div", "pruning-pair-list"), lookup = new Map((family.members || []).map(v => [idOf(v), v]));
            for (const proposal of family.proposals || []) {
                const candidate = lookup.get(proposal.candidate_id), witness = lookup.get(proposal.witness_id) || proposal.scene_witness; if (!candidate || !witness) continue;
                const row = h("article", "pruning-pair"), images = h("div", "pruning-pair-images"); images.append(thumbnail(candidate, context, "Proposed removal"), thumbnail(witness, context, "Keep witness"));
                const evidence = proposal.evidence || {}; row.append(h("strong", "", `${imageLabel(proposal.candidate_id)} → ${imageLabel(proposal.witness_id)} · ${pct(evidence.combined)}`), images);
                const mode = state.result?.resolved_config?.evidence_mode || state.result?.config?.evidence_mode || state.result?.recipe?.evidence_mode || "compound";
                row.append(h("p", "", budgeted ? "This picture is a retained scene witness. Individual objects can match different retained pictures; uncovered appearances count against both loss ceilings. Inspect the Impact view for those object witnesses and losses." : `${mode === "scene_only" ? "Label inventory is covered; bbox appearance is unmeasured" : `Weakest candidate object ${pct(evidence.minimum_object_appearance)} · objects ${pct(evidence.object)}`} · scene ${pct(evidence.scene)}. ${evidence.exact_image ? "Identical encoded image bytes. " : ""}${candidate.objects?.length ? mode === "scene_only" ? "Candidate classes and counts must be present in the keeper; appearance is not verified." : "Every candidate object needs a same-class assignment; the keeper may have extra objects." : "Explicit negative: scene evidence only."}`));
                row.append(h("p", `pruning-badge ${proposal.status === "approved" ? "good" : ""}`, proposal.status === "approved" ? `Approved to remove ${imageLabel(proposal.candidate_id)}` : proposal.status === "rejected" ? "This witness rejected" : "Unreviewed · retained in export"));
                const actions = h("div", "pruning-inline"), noteDetails = h("details"), note = h("textarea"); note.rows = 2; note.placeholder = "Reason or note (optional)"; const savedDecision = state.review?.decisions?.[proposal.candidate_id]; note.value = state.noteDrafts.get(`${proposal.candidate_id}:${proposal.evidence_digest}`) ?? proposal.note ?? (savedDecision?.evidence_digest === proposal.evidence_digest ? savedDecision.note : "") ?? ""; note.addEventListener("input", () => { state.noteDrafts.set(`${proposal.candidate_id}:${proposal.evidence_digest}`, note.value); remember(); }); noteDetails.append(h("summary", "pruning-help", "Reason / note"), note);
                const approved = button(`OK to remove ${imageLabel(proposal.candidate_id)}`, () => reviewAction({ action: "approve", candidate_id: proposal.candidate_id, witness_id: proposal.witness_id, evidence_digest: proposal.evidence_digest, note: note.value }), "pruning-approve"); approved.dataset.reviewAction = ""; approved.dataset.requiresConfirmation = String(!budgeted && !family.confirmed);
                const keep = button(`Keep ${imageLabel(proposal.candidate_id)} too`, () => reviewAction({ action: "pin", image_id: proposal.candidate_id, keep: true, note: note.value }), "pruning-keep"); keep.dataset.reviewAction = "";
                const reject = button("Reject this match", () => reviewAction({ action: "reject", candidate_id: proposal.candidate_id, witness_id: proposal.witness_id, evidence_digest: proposal.evidence_digest, note: note.value })); reject.dataset.reviewAction = "";
                const undo = button("Undo decision", () => reviewAction({ action: "undo_decision", candidate_id: proposal.candidate_id })); undo.dataset.reviewAction = "";
                actions.append(approved, keep); if (!budgeted) actions.append(reject); actions.append(undo, button(budgeted ? "Inspect object coverage" : "Compare exact bboxes", () => budgeted ? objectCoverageDialog({ ...proposal, candidate, scene_witness: witness, object_witnesses: evidence.object_witnesses || evidence.objects || [] }) : comparison(proposal.candidate_id, proposal.witness_id)));
                if (savedDecision?.note && savedDecision.evidence_digest !== proposal.evidence_digest) row.append(h("p", "pruning-help", `Previous evidence note (history): ${savedDecision.note}`));
                row.append(actions, h("p", "pruning-help", budgeted ? "Keep adds an independent image pin and restores its coverage without replacing it with a new proposal. Approve binds this exact candidate and its measured policy evidence." : `Both choices retain the witness ${imageLabel(proposal.witness_id)}. Keep adds an independent image pin; Reject only rejects this directed match. ${family.confirmed ? "" : "Confirm keepers before individual approval, or use the bulk preview."}`), noteDetails); pairs.append(row);
            }
            body.append(pairs);
            if (family.member_next_offset !== null && family.member_next_offset !== undefined) {
                body.append(h("p", "pruning-help", `Showing ${number(family.members.length)} of ${number(total)} members. Load more to inspect the remaining images and exact proposals. Keep-X and bulk previews use the complete family.`));
                const more = button("Load more family images", () => loadMembers(familyId, context)); more.dataset.reviewAction = ""; body.append(more);
            }
            card.append(heading, body);
            if (old) { old.replaceWith(card); Array.from(card.querySelectorAll("details")).forEach((n, i) => { n.open = !!detailsOpen[i]; }); } else el.review.append(card);
            el.review.append(card); state.familyNodes.set(familyId, card); return card;
        }
        async function loadMembers(familyId, context) {
            const previewId = planningDirty() ? state.planning.preview?.preview_id : null;
            const sourcePlan = previewId ? state.planning.preview.plan : state.plan;
            const family = state.families.get(familyId), jobId = state.jobId, generation = sourcePlan.generation, displayGeneration = state.generation;
            if (!family || family.member_next_offset == null || state.busy || state.pending || state.reviewPending) return;
            state.reviewPending = true; controls();
            try {
                const page = await request(`/jobs/${encodeURIComponent(jobId)}/families/${encodeURIComponent(familyId)}/members`, post({ ...reviewPayload(), ...(previewId ? { preview_id: previewId } : {}), generation, offset: family.member_next_offset, limit: 120 }), context);
                if (state.jobId !== jobId || displayGeneration !== state.generation || previewId !== (planningDirty() ? state.planning.preview?.preview_id : null)) return;
                const known = new Set(family.members.map(idOf)); family.members.push(...page.members.filter(member => !known.has(idOf(member))));
                family.member_next_offset = page.next_offset; family.members_truncated = page.next_offset !== null;
                renderFamily(family, context);
            } catch (error) { handleError(error, context); }
            finally { if (displayGeneration === state.generation && previewId === (planningDirty() ? state.planning.preview?.preview_id : null)) { state.reviewPending = false; controls(); } }
        }
        async function families(context, reset = false) {
            if (!state.result || !current(context)) return;
            const jobId = state.jobId, serial = reset ? ++state.serial : state.serial, settingsPreviewId = planningDirty() ? state.planning.preview?.preview_id : null;
            if (planningDirty() && state.planning.preview?.status !== "ready") return;
            if (state.reviewPending && !reset && el.review.getAttribute("aria-busy") === "true") return;
            state.reviewPending = true; controls(); el.review.setAttribute("aria-busy", "true");
            const offset = reset ? 0 : state.next;
            if (offset === null) { state.reviewPending = false; controls(); return; }
            const selection = { ...reviewPayload(), ...(settingsPreviewId ? { preview_id: settingsPreviewId } : {}) }, query = el.search.value.trim(), filter = el.filter.value, sort = el.sort.value;
            const listSignature = digest([query, filter, sort]), preserveCount = reset && state.listSignature === listSignature ? state.offset : 0;
            const anchorNode = reset ? Array.from(state.familyNodes.values()).find(n => n.getBoundingClientRect().bottom > 0 && n.getBoundingClientRect().top < root.innerHeight) : null;
            const anchor = anchorNode ? { id: anchorNode.dataset.familyId, top: anchorNode.getBoundingClientRect().top } : null;
            try {
                let data = await request(`/jobs/${encodeURIComponent(jobId)}/families`, post({ ...selection, offset, limit: 12, ...(reset ? {} : { plan_revision: state.plan?.plan_revision }), query, filter, sort }), context);
                if (reset && preserveCount > data.families.length) {
                    const populationGeneration = digest(data.generation), rows = data.families.slice(); let cursor = data.next_offset;
                    while (rows.length < preserveCount && cursor !== null && cursor !== undefined) {
                        const page = await request(`/jobs/${encodeURIComponent(jobId)}/families`, post({ ...selection, expected_review_revision: data.review_revision, offset: cursor, limit: 12, plan_revision: data.plan_revision, query, filter, sort }), context);
                        if (serial !== state.serial || !current(context) || state.jobId !== jobId) return;
                        if (digest(page.generation) !== populationGeneration) throw new Error("Family generation changed during refresh. Reconnect the current review.");
                        rows.push(...page.families); cursor = page.next_offset;
                    }
                    data = { ...data, families: rows, next_offset: cursor };
                }
                if (serial !== state.serial || !current(context) || state.jobId !== jobId || settingsPreviewId !== (planningDirty() ? state.planning.preview?.preview_id : null)) return;
                const generation = digest(data.generation || [selection.selection_revision, data.plan_revision, data.plan_digest, data.review_revision]);
                if (!reset && generation !== state.generation) { state.reviewPending = false; await families(context, true); return; }
                if (reset) {
                    state.generation = generation; state.listSignature = listSignature; state.families.clear();
                    const incoming = new Set((data.families || []).map(v => v.id));
                    for (const [id, node] of state.familyNodes) if (!incoming.has(id)) { node.remove(); state.familyNodes.delete(id); }
                    if (!settingsPreviewId) state.plan = data;
                    state.filteredFamilyIds = data.filtered_family_ids || null;
                    for (const draft of state.familyDrafts.values()) draft.preview = null;
                }
                state.next = data.next_offset ?? null; state.offset = offset + (data.families || []).length;
                for (const family of data.families || []) { state.families.set(family.id, family); renderFamily(family, context); }
                if (!settingsPreviewId) {
                    state.plan = { ...state.plan, removed_count: data.removed_count, retained_count: data.retained_count, approved_count: data.approved_count, plan_revision: data.plan_revision, plan_digest: data.plan_digest, review_revision: data.review_revision, coverage: data.coverage || state.plan?.coverage, ...Object.fromEntries(["proposed_loss", "approved_loss", "appearance_group_count", "total_appearance_weight", "curve"].filter(key => data[key] !== undefined).map(key => [key, data[key]])) };
                    state.review = { ...state.review, review_revision: data.review_revision, plan_revision: data.plan_revision, pins: data.pins, decisions: data.decisions, constraints: data.constraints, batches: data.batches };
                } else state.planning.preview.plan = { ...state.planning.preview.plan, ...data };
                if (reset && !data.families?.length) el.review.replaceChildren(h("p", "pruning-help", "No matching families at this removal ceiling and filter. Unreviewed images remain in the dataset."));
                el.page.textContent = `${number(data.filtered_family_count ?? data.family_count)} families · ${number(data.removed_count)} proposals · ${number(data.approved_count)} approved`;
                el.sentinel.textContent = state.next === null ? "All matching families loaded." : "Scroll to load the next families.";
                const splits = data.splits; if (splits) el.split.textContent = `${number(splits.train_count)} train / ${number(splits.val_count)} validation · ${number(splits.group_count)} related groups. ${(splits.warnings || []).join(" ")}`;
                state.reviewPending = false; state.recoveryRequired = false; el.review.setAttribute("aria-busy", "false"); render();
                if (anchor && state.listSignature === listSignature) { const freshAnchor = state.familyNodes.get(anchor.id); if (freshAnchor) root.scrollBy(0, freshAnchor.getBoundingClientRect().top - anchor.top); }
            } catch (error) {
                if (serial !== state.serial || !current(context)) return;
                state.reviewPending = false; state.recoveryRequired = true; el.review.setAttribute("aria-busy", "false"); el.sentinel.textContent = "Family loading paused. Reconnect or load more to retry."; controls(); throw error;
            } finally {
                if (serial === state.serial && current(context) && jobId === state.jobId) { state.reviewPending = false; el.review.setAttribute("aria-busy", "false"); controls(); }
            }
        }
        async function familyPreview(familyId, delayed = false) {
            const draft = state.familyDrafts.get(familyId), context = state.context, jobId = state.jobId;
            if (!draft || state.busy || state.exporting || state.pending || planningDirty()) return;
            const serial = ++draft.serial, generation = state.generation;
            const doPreview = async () => {
                if (!current(context) || generation !== state.generation || serial !== draft.serial || jobId !== state.jobId) return;
                const status = state.familyNodes.get(familyId)?.querySelector('[data-family-preview-status]'); if (status) status.textContent = "Computing direct coverage and required keepers …";
                try {
                    const preview = await request(`/jobs/${encodeURIComponent(jobId)}/plan/preview`, post({ ...reviewPayload(), family_id: familyId, family_ids: [familyId], keep_count: draft.count, ...(draft.keeperIds ? { keeper_ids: draft.keeperIds } : {}) }), context);
                    if (!current(context) || generation !== state.generation || serial !== draft.serial || jobId !== state.jobId) return;
                    draft.preview = preview;
                    if (status) status.textContent = `${number(preview.captured_eligible_count ?? preview.removed_count)} eligible removals in this family · ${number(preview.removed_count)} proposed globally · ${number(preview.additional_keeper_count)} additional keeper requirements. ${(preview.exceptions || []).map(v => typeof v === "string" ? v : (v.reasons || [v.reason || "Coverage/protection exception"]).join(" ")).join(" ")}`;
                    const confirm = state.familyNodes.get(familyId)?.querySelector('[data-family-confirm]'); if (confirm) { confirm.dataset.requiresConfirmation = "false"; confirm.disabled = false; }
                } catch (e) { if (current(context) && serial === draft.serial) { if (status) status.textContent = e.message; handleError(e, context); } }
            };
            clearTimeout(draft.timer); if (delayed) draft.timer = setTimeout(doPreview, 220); else await doPreview();
        }
        async function confirmFamily(familyId) {
            const draft = state.familyDrafts.get(familyId); if (!draft?.preview) return;
            await reviewAction({ action: "confirm_plan", preview_digest: draft.preview.preview_digest });
        }
        function dialog(kind, title) {
            if (state.dialog?.open) state.dialog.close();
            state.dialog?.remove(); const n = h("dialog", "pruning-dialog"); n.dataset.kind = kind;
            const heading = h("div", "pruning-dialog-heading"), headingTitle = h("h3", "", title); headingTitle.id = `pruningDialog${nonce()}`;
            n.setAttribute("aria-labelledby", headingTitle.id); heading.append(headingTitle, button("Close", () => n.close())); n.append(heading);
            document.body.append(n); state.dialog = n; state.dialogContext = state.context;
            n.addEventListener("close", () => { if (state.dialog === n) state.dialogContext = null; });
            n.addEventListener("click", event => { if (event.target === n && !state.busy) { const r = n.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) n.close(); } });
            if (typeof n.showModal === "function") n.showModal(); else { n.setAttribute("open", ""); n.setAttribute("role", "dialog"); }
            return n;
        }
        async function filteredIds(context) {
            if (state.filteredFamilyIds) return state.filteredFamilyIds.slice();
            const jobId = state.jobId, generation = state.generation; const values = [];
            let offset = 0;
            do {
                const data = await request(`/jobs/${encodeURIComponent(jobId)}/families`, post({ ...payload(), offset, limit: 12, plan_revision: state.plan.plan_revision, expected_review_revision: state.plan.review_revision, query: el.search.value.trim(), filter: el.filter.value, sort: el.sort.value }), context);
                if (generation !== state.generation || digest(data.generation || [state.result.selection_revision, data.plan_revision, data.plan_digest, data.review_revision]) !== generation) throw new Error("The family population changed. Refresh the list before previewing this scope.");
                values.push(...(data.families || []).map(v => v.id)); offset = data.next_offset;
            } while (offset !== null && offset !== undefined);
            return values;
        }
        async function bulkPreview() {
            if (state.busy || state.exporting || state.externalRefreshing || state.reviewPending || state.pending || planningDirty() || !state.result) return;
            const context = state.context, jobId = state.jobId, generation = state.generation; const operation = ++state.operation;
            state.busy = true; controls(); message("Capturing the exact family scope and computing a concrete keeper plan …");
            try {
                const scope = el.bulkScope.value, familyIds = scope === "filtered" ? await filteredIds(context) : null;
                const keepCount = Math.max(1, Math.round(Number(el.bulkCount.value) || 1));
                const value = await request(`/jobs/${encodeURIComponent(jobId)}/plan/preview`, post({ ...reviewPayload(), family_ids: familyIds, keep_count: keepCount }), context);
                if (!current(context) || generation !== state.generation || operation !== state.operation) return;
                state.preview = { ...value, context_signature: signature(context), job_id: jobId, generation, scope_label: scope === "all" ? "All analysis families" : `Current filter (${el.search.value || "all names"}, ${el.filter.selectedOptions[0].textContent})` };
                showPreview(state.preview, context); message("Preview ready. No removals have been approved by this preview.");
            } catch (e) { handleError(e, context); }
            finally { if (operation === state.operation) { state.busy = false; controls(); if (state.externalRefreshNeeded) refreshExternal(state.context); } }
        }
        function showPreview(preview, context) {
            const n = dialog("preview", "Keep X and approve eligible removals"), stats = h("div", "pruning-preview-stats");
            n.append(h("p", "", `${preview.scope_label} · ${number(preview.family_ids?.length)} captured families · keep at least ${number(preview.requested_keep_count || el.bulkCount.value)} images where feasible. Pins, useful appearance/scale coverage and the global ceiling can require additional keepers.`));
            for (const [value, label] of [[preview.captured_eligible_count ?? "—", "Eligible in captured families"], [preview.captured_retained_count ?? "—", "Retained in captured families"], [preview.new_approval_count, "New approvals"], [preview.additional_keeper_count, "Additional keeper requirements"]]) { const cell = h("div", "pruning-preview-stat"); cell.append(h("strong", "", typeof value === "number" ? number(value) : "Not supplied"), h("span", "", label)); stats.append(cell); } n.append(stats);
            n.append(h("p", "", `Global plan: ${number(preview.removed_count)} proposals and ${number(preview.retained_count)} retained of ${number(state.result.image_count)} images (${Number(preview.removed_percent ?? (100 * preview.removed_count / state.result.image_count)).toFixed(2)}% proposed removal) · ${number(preview.superseded_count)} existing approvals superseded by changed evidence/witnesses. Only the captured eligible set is approved. Future results and out-of-scope new proposals remain unapproved.`));
            const exceptions = h("details", "pruning-disclosure"); exceptions.append(h("summary", "", `${number(preview.exceptions?.length)} family exceptions / extra keepers`));
            for (const item of preview.exceptions || []) { const p = h("p", "pruning-help", typeof item === "string" ? item : item.reason || item.reasons?.join(" ") || JSON.stringify(item)); if (item.family_id) p.append(button("Open family", () => { n.close(); locateFamily(item.family_id); })); exceptions.append(p); } n.append(exceptions);
            const impact = h("details", "pruning-disclosure"); impact.append(h("summary", "", "Per-class retained annotation impact"));
            const rows = Array.isArray(preview.class_impact) ? preview.class_impact : Object.entries(preview.class_impact || {}).map(([class_id, value]) => ({ class_id, ...(typeof value === "object" ? value : { retained: value }) }));
            for (const value of rows) impact.append(h("p", "pruning-help", `${value.class_name || className(value.class_id)} · source ${number(value.source ?? value.source_count)} · retained ${number(value.retained ?? value.retained_count)} · removed ${number(value.removed ?? value.removed_count ?? ((value.source ?? value.source_count) - (value.retained ?? value.retained_count)))}`));
            if (!rows.length) impact.append(h("p", "pruning-help", "Class impact must be present in the authoritative export manifest; no class-balance benefit is inferred here.")); n.append(impact);
            const provenance = h("details", "pruning-disclosure"); provenance.append(h("summary", "", "Source and recipe binding"), h("p", "pruning-help", `${context.datasetName || sourceKey(context)} · saved source revision ${preview.source_revision} · recipe ${preview.recipe_digest} · plan revision ${preview.plan_revision} · review revision ${preview.review_revision} · preview ${preview.preview_digest}`)); n.append(provenance);
            const status = h("p", "pruning-status"); status.dataset.previewStatus = "";
            const commit = button("Apply and approve eligible removals", async () => { if (state.preview !== preview || !current(context) || state.jobId !== preview.job_id || state.generation !== preview.generation) { status.textContent = "Preview is stale. Generate a fresh preview."; commit.disabled = true; return; } const committed = await reviewAction({ action: "bulk_commit", preview_digest: preview.preview_digest }); if (committed && current(context) && !state.recoveryRequired && !state.pending) n.close(); }, "pruning-primary"); commit.dataset.previewCommit = ""; commit.dataset.mutation = "";
            const actions = h("div", "pruning-inline"); actions.append(commit, button("Continue reviewing", () => n.close())); n.append(status, actions, h("p", "pruning-help", "This approves one exact review plan. Download variant creates the new artifact afterwards; source images and annotations are unchanged."));
        }
        async function locateFamily(id) {
            if (state.busy || state.exporting || state.externalRefreshing || state.pending || state.recoveryRequired) return;
            let n = state.familyNodes.get(id);
            if (!n) { el.search.value = id; el.filter.value = "all"; await families(state.context, true); n = state.familyNodes.get(id); }
            n?.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        async function refreshReview(context) {
            const jobId = state.jobId;
            const snapshot = await request(`/jobs/${encodeURIComponent(jobId)}/review`, {}, context);
            if (state.jobId !== jobId) return;
            state.review = snapshot; state.lastBatch = lastBatch(snapshot); loaded(await request(`/jobs/${encodeURIComponent(jobId)}/result`, {}, context)); syncPlanning(true);
            await families(context, true); if (state.planning.view === "impact") await impactCards(context, true);
        }
        async function reviewAction(value) {
            if (state.busy || state.exporting || state.externalRefreshing || state.reviewPending || state.pending || state.recoveryRequired || (planningDirty() && value.action !== "apply_plan") || !state.result) return;
            const operation = ++state.operation, context = state.context, jobId = state.jobId;
            state.busy = true; state.serial++; clearTimeout(state.timer); controls(); message("Saving the exact review change …");
            let committed = false;
            try {
                const answer = await mutation("review", `/jobs/${encodeURIComponent(jobId)}/review`, { ...reviewPayload(), ...value }, context);
                committed = true; state.review = answer.snapshot || answer.review || state.review;
                if (value.action === "bulk_commit") state.lastBatch = answer.receipt?.batch_id || answer.receipt?.id || answer.snapshot?.last_batch_id || "";
                if (value.action === "bulk_undo") state.lastBatch = lastBatch(answer.snapshot);
                if (value.action === "apply_plan") state.planning.preview = null;
                resetPlanning(); invalidateExport(); invalidatePreview(); await refreshReview(context); remember();
                message(answer.message || (value.action === "apply_plan" ? "Exact preview applied. No removals were approved. Review the impact cards before preparing a variant." : value.action === "pin" ? "Keep choice saved. The captured proposals were reduced without filling the gap. Recalculate proposals when you want a new plan." : value.action === "bulk_commit" ? "Keeper plan and exact eligible approvals saved as one batch. Download the reviewed variant when ready." : value.action === "bulk_undo" ? "Batch-owned decisions undone. Independent later Keep choices and decisions remain." : "Review saved. Current keepers, witnesses and approvals refreshed from the server."));
                return true;
            } catch (e) {
                if (!current(context) || e.name === "AbortError") return;
                if (committed) { state.recoveryRequired = true; message("Review saved, but the view did not refresh. Reconnect saved review; the server receipt remains authoritative.", true); }
                else { if (!state.pending && e.status === 409) { resetPlanning(); try { await refreshReview(context); } catch (_) { state.recoveryRequired = true; } } message(e.message, true); }
            } finally { if (operation === state.operation) { state.busy = false; controls(); if (state.externalRefreshNeeded) refreshExternal(state.context); } }
        }
        function memberDetails(family, image) {
            const n = dialog("member", `Family image ${imageLabel(idOf(image))}`); n.append(thumbnail(image, state.context, "Source image"));
            const area = (image.objects || []).map(v => `${className(v.class_id)} row ${v.label_row ?? v.row ?? "recorded"}: relative area ${pct(v.box?.[2] * v.box?.[3])}, bbox approximately ${Math.round((v.box?.[2] || 0) * image.width)} × ${Math.round((v.box?.[3] || 0) * image.height)} source pixels`).join("; ");
            n.append(h("p", "", `Original image ${image.width} × ${image.height} pixels. ${area || "Explicit negative or protected annotation state."}`));
            if (state.review?.pins?.[idOf(image)]?.note) n.append(h("p", "pruning-help", `Independent Keep note: ${state.review.pins[idOf(image)].note}`));
            appendHistory(n, idOf(image));
            n.append(h("p", "pruning-help", "Original dimensions and object occupancy remain useful protection evidence even when size is corrected out of similarity. Detail/focus is a proxy, not a pose or training-value judgment."));
            const other = select((family.members || []).filter(v => idOf(v) !== idOf(image)).map(v => [idOf(v), v.image_relpath || idOf(v)]));
            n.append(field("Compare this candidate with", other), button("Compare exact bboxes", () => comparison(idOf(image), other.value)));
        }
        function appendHistory(container, imageId) {
            const entries = (state.review?.history || []).filter(value => value.candidate_id === imageId || value.image_id === imageId || value.touched_fields?.some(field => field.key === imageId));
            if (!entries.length) return;
            const details = h("details", "pruning-disclosure"); details.append(h("summary", "", `Saved decision history · ${entries.length} changes`));
            for (const entry of entries) {
                const notes = [...new Set([entry.note, entry.previous_decision?.note, ...(entry.touched_fields || []).filter(field => field.key === imageId).map(field => field.before?.note)].filter(Boolean))];
                details.append(h("p", "pruning-help", `Revision ${entry.revision} · ${entry.action}${notes.length ? ` · ${notes.join(" / ")}` : ""}`));
            }
            container.append(details);
        }
        function assetURL(value, context) {
            if (!value) return "";
            if (/^https?:\/\//.test(value) || value.startsWith("data:")) return value;
            const base = new URL(context.apiRoot, root.location.href);
            return value.startsWith("/dataset_pruning/") ? `${context.apiRoot}${value}` : value.startsWith("/") ? new URL(value, base.origin).href : `${context.apiRoot}/${value}`;
        }
        function objectCoverageDialog(row) {
            const context = state.context, jobId = state.jobId, candidate = row.candidate;
            if (!candidate || !current(context)) return;
            const candidateId = idOf(candidate), n = dialog("object-coverage", `${imageLabel(candidateId)} · retained object coverage`);
            n.append(h("p", "", "The scene witness preserves a similar whole-picture context. It does not have to contain every object. Each original object is checked for a qualifying same-class, original-size-compatible match anywhere in the final retained dataset; different objects can use different pictures. One retained appearance can cover several original boxes. This measures appearance diversity, not physical object counts."));
            n.append(h("p", "pruning-help", `${planningDirty() ? "Preview" : "Applied plan"} · source revision ${state.result.source_revision ?? context.revision} · plan revision ${state.review?.plan_revision ?? state.plan?.plan_revision}. These are the recorded direct object witnesses, not a strict one-to-one comparison against the scene picture.`));
            const frames = h("div", "pruning-comparison-images"); frames.append(thumbnail(candidate, context, "Candidate image", true));
            if (row.scene_witness) frames.append(thumbnail(row.scene_witness, context, "Retained scene / context witness", true)); n.append(frames);
            const objects = row.object_witnesses || [], grid = h("div", "pruning-crop-grid");
            function nativeCrop(imageId, objectRow, caption) {
                const figure = h("figure"); figure.style.margin = "0";
                if (Number.isInteger(objectRow) && objectRow >= 0) {
                    const img = h("img"); img.loading = "lazy"; img.decoding = "async"; img.alt = caption;
                    img.src = `${context.apiRoot}/dataset_pruning/jobs/${encodeURIComponent(jobId)}/images/${encodeURIComponent(imageId)}/objects/${objectRow}`;
                    figure.append(img);
                }
                figure.append(h("figcaption", "pruning-help", caption)); return figure;
            }
            for (const object of objects) {
                const objectRow = object.row ?? object.label_row ?? object.candidate_row;
                const witnessId = object.witness_image_id || object.witness_id;
                const witnessRow = object.witness_row;
                const covered = object.covered === true && !!witnessId;
                const item = h("article", "pruning-crop-row"), pair = h("div", "pruning-crop-pair");
                item.append(h("strong", "", `${className(object.class_id)} · candidate row ${objectRow}`));
                pair.append(nativeCrop(candidateId, objectRow, "Candidate · original object cutout"));
                if (covered && Number.isInteger(witnessRow) && witnessRow >= 0) pair.append(nativeCrop(witnessId, witnessRow, `Retained match · ${imageLabel(witnessId)} · row ${witnessRow}`));
                else if (covered) pair.append(h("p", "pruning-help", `A qualifying retained match is verified in ${imageLabel(witnessId)}. This record has no exact witness-row cutout; no substitute crop is shown.`));
                else pair.append(h("p", object.unknown ? "pruning-help" : "pruning-gap", object.unknown ? "Coverage is not resolved in this record. This is not evidence of absence." : "No qualifying retained object match. This original appearance counts toward the plan's loss accounting."));
                item.append(pair, h("p", "pruning-help", covered ? `Covered by a retained object${finiteNumber(object.similarity ?? object.appearance) ? ` · appearance ${pct(object.similarity ?? object.appearance)}` : ""}. The scene witness may be a different image.` : object.unknown ? "Unknown direct coverage." : "Uncovered · no retained object witness."));
                grid.append(item);
            }
            if (!objects.length) grid.append(h("p", "pruning-help", candidate.objects?.length ? "This card has no per-object witness records. Do not interpret that as zero object loss; use the measured plan totals." : "This image has no labelled objects. Its scene witness must satisfy the explicit-negative scene rule."));
            n.append(grid, h("p", "pruning-help", "Keeping this image can restore some original objects without fully restoring an affected appearance group. Approval and export remain separate decisions."));
        }
        async function comparison(candidateId, witnessId) {
            const context = state.context, jobId = state.jobId, selectionRevision = state.result.selection_revision;
            const n = dialog("comparison", `${imageLabel(candidateId)} → ${imageLabel(witnessId)} · exact object comparison`), status = h("p", "pruning-status", "Loading source-bound whole images and bbox assignments …"); n.append(status);
            try {
                const data = await request(`/jobs/${encodeURIComponent(jobId)}/comparison`, post({ selection_revision: selectionRevision, candidate_id: candidateId, witness_id: witnessId }), context);
                if (!current(context) || !n.open || state.dialog !== n || jobId !== state.jobId) return;
                status.textContent = state.planning.active?.policy === "budgeted_coverage" ? "This inspects one scene-witness pair. Global object coverage may use several other retained images; inspect Impact for those witnesses and permitted losses. Close fingerprints do not prove physical identity." : "Candidate-side coverage is directional. Close descriptor matches do not establish physical instance identity.";
                const images = h("div", "pruning-comparison-images"); images.append(thumbnail(data.candidate, context, "Candidate / proposed removal"), thumbnail(data.witness, context, "Witness / retained image")); n.append(images);
                const addCalibration = button("Add this pair to recipe cutoff review", () => { n.close(); recipeLab?.addPair(candidateId, witnessId); }); addCalibration.dataset.mutation = ""; n.append(addCalibration);
                appendHistory(n, candidateId);
                const e = data.evidence || {}, matches = e.matches || e.forward?.matches || [];
                const candidateSide = e.reverse === true ? "right" : "left", oppositeSide = e.reverse === true ? "left" : "right";
                const sceneScore = e.scene_evidence_used === false ? null : e.scene ?? e.forward?.scene;
                const objectScore = e.object_evidence_used === false ? null : e.object ?? e.forward?.object;
                const weakest = e.object_evidence_used === false ? null : e.minimum_object_appearance ?? e[`${candidateSide}_minimum_object_appearance`] ?? e.forward?.minimum_object_appearance;
                n.append(h("p", "", `Index ${pct(e.combined ?? e.forward?.combined)} · scene ${pct(sceneScore)} · object coverage ${pct(objectScore)} · weakest candidate object ${pct(weakest)}. ${(data.candidate.objects || []).length} candidate objects; ${(data.witness.objects || []).length} witness objects.`));
                const crops = h("div", "pruning-crop-grid");
                function maskDescription(value, side) {
                    if (value?.status === "not_used") return `${side}: SAM not used by this recipe.`;
                    if (value?.status !== "known") return `${side}: per-object mask evidence unknown. ${value?.reason || "Historical aggregates cannot identify this bbox."}`;
                    return `${side}: ${value.used_bbox_fallback ? "bbox pooling fallback" : "SAM mask pooling"}${value.mask_score == null ? "" : ` · segmentation mask score ${pct(value.mask_score)}`}.`;
                }
                for (const crop of data.crops || []) {
                    const item = h("div", "pruning-crop-row"), pair = h("div", "pruning-crop-pair");
                    for (const [url, label] of [[crop.candidate_url, "Candidate bbox"], [crop.witness_url, "Witness bbox"]]) { const img = h("img"); img.loading = "lazy"; img.src = assetURL(url, context); img.alt = `${label}: ${className(crop.class_id)}`; pair.append(img); }
                    item.append(h("strong", "", className(crop.class_id)), pair, h("p", "pruning-help", `Rows ${crop.candidate_row} → ${crop.witness_row} · appearance ${pct(crop.appearance)} · scale ${pct(crop.scale)}. ${maskDescription(crop.candidate_mask_evidence, "Candidate")} ${maskDescription(crop.witness_mask_evidence, "Witness")}`)); crops.append(item);
                }
                n.append(crops);
                const table = h("table", "pruning-components-table"), header = h("tr"); for (const value of ["Class", "Candidate → witness rows", "Appearance", "Scale / geometry", "Accepted"]) header.append(h("th", "", value)); table.append(header);
                for (const match of matches) {
                    const candidateRow = match.candidate_row ?? (e.reverse === true ? match.right_row : match.left_row);
                    const witnessRow = match.witness_row ?? (e.reverse === true ? match.left_row : match.right_row);
                    const row = h("tr"); for (const v of [className(match.class_id), `${candidateRow} → ${witnessRow}`, pct(match.appearance), `${pct(match.scale)} / ${pct(match.geometry)}`, match.accepted === false ? "No" : "Yes"]) row.append(h("td", "", v)); table.append(row);
                }
                n.append(table);
                if (typeof e[`${oppositeSide}_valid`] === "boolean") n.append(h("p", "pruning-help", `Witness → candidate direction: ${e[`${oppositeSide}_valid`] ? "eligible" : "not eligible"} under this recipe · weakest witness object ${pct(e.object_evidence_used === false ? null : e[`${oppositeSide}_minimum_object_appearance`])}. Extra witness objects can make the directions different.`));
                const details = h("details", "pruning-disclosure"); details.append(h("summary", "", "Exact evidence / recipe binding"), h("pre", "pruning-help", JSON.stringify({ evidence_digest: data.evidence_digest, evidence: data.evidence }, null, 2))); n.append(details);
            } catch (e) { if (current(context) && state.dialog === n && e.name !== "AbortError") { status.textContent = e.message; status.classList.add("pruning-error"); } }
        }
        async function verifyBackend() {
            try { await adapters.verifyBackend?.(); return true; }
            catch (error) {
                if (error.name === "AbortError") return false;
                begin(adapters.context()); message(error.message, true); return false;
            }
        }
        async function openProfiles(refreshOf = null) {
            const stillRequested = () => !refreshOf || (state.dialog === refreshOf && refreshOf.open);
            if (!await verifyBackend()) return;
            if (!stillRequested()) return;
            if (!state.initialized && !state.busy && !state.exporting) await restore();
            if (!stillRequested()) return;
            const context = begin(adapters.context());
            const target = adapters.profileContext?.() || context.annotationSource; const identity = target?.kind && target?.id ? `${target.kind}:${target.id}` : sourceKey(context);
            if (!context.apiRoot || !identity) { message("Choose a dataset on a connected backend before managing its profiles.", true); return; }
            const n = dialog("profiles", "Dataset SALAD profiles"), status = h("p", "pruning-status", "Loading the shared profile registry …"); n.append(status); n.dataset.sourceLabel = target.label || target.name || (identity === sourceKey(context) ? context.datasetName : identity) || identity;
            n.dataset.sourceIdentity = identity;
            try {
                if (!state.capabilities) await capabilities(context);
                const snapshot = await readProfiles(context, identity);
                if (!current(context) || state.dialog !== n || !n.open) return;
                renderProfiles(n, snapshot, context, identity);
            } catch (e) { if (current(context) && state.dialog === n) { status.textContent = e.message; status.classList.add("pruning-error"); } }
        }
        function renderProfiles(n, snapshot, context, identity) {
            const heading = n.firstElementChild; n.replaceChildren(heading);
            const dataset = snapshot.dataset || snapshot.datasets?.[identity] || { head_ids: [], preferred_recipes: {} }, attached = new Set(dataset.head_ids || []);
            n.append(h("p", "", `${n.dataset.sourceLabel || identity} · ${number(attached.size)} attached heads · management revision ${snapshot.revision}. These are deployment associations. A profile's historical training reference is unchanged.`));
            const table = h("table", "pruning-profile-table"), rowHead = h("tr"); for (const title of ["Profile / scope", "Backbone / training reference", "Dataset association", "Management"]) rowHead.append(h("th", "", title)); table.append(rowHead);
            for (const head of heads()) {
                const record = snapshot.profiles?.[head.id] || {}, archived = Boolean(record.archived || head.archived), row = h("tr");
                const nameCell = h("td"), title = h("strong", "", record.display_label || head.display_label || head.label || head.id);
                nameCell.append(title, h("p", "pruning-help", `Immutable head ${head.id} · ${headScope(head)} · ${head.descriptor_dim || head.descriptor_dimension || "saved"}D${archived ? " · archived from selection" : ""}`));
                const inspect = h("details"); inspect.append(h("summary", "", "Training and compatibility"), h("pre", "pruning-help", JSON.stringify({ ...head, display_label: record.display_label || head.label, deployment_dataset: identity, attached: attached.has(head.id) }, null, 2))); nameCell.append(inspect);
                const training = h("td"); training.append(h("strong", "", `${head.encoder_type || "recorded encoder"} · ${head.encoder_model || "recorded model"}`), h("p", "pruning-help", `${number(head.train_image_count)} training images/objects · ${head.reference_label || head.reference_dataset_label || "historical reference"} · ${head.created_at ? new Date(Number(head.created_at) * (Number(head.created_at) < 1e12 ? 1000 : 1)).toLocaleDateString() : "recorded training date"}`));
                const referenceMatches = head.reference_dataset_id === context.annotationSource?.id || head.reference_session_id === context.annotationSource?.id;
                training.append(h("p", "pruning-help", referenceMatches ? "Associated training source ID; freshness requires the recorded content fingerprint, not the name or image count." : "Trained on another or older recorded snapshot. Attachment does not retrain or rewrite provenance; compatible heads can still be used."));
                const association = h("td"), attach = button(attached.has(head.id) ? "Detach from dataset" : "Attach to dataset", () => manageProfile({ action: attached.has(head.id) ? "detach" : "attach", head_id: head.id, role: headScope(head) }, snapshot, context, identity, n)); attach.dataset.mutation = "";
                association.append(h("p", "", attached.has(head.id) ? "Attached" : "Not attached"), attach);
                const actions = h("td"), actionRow = h("div", "pruning-profile-actions"), rename = input("text", record.display_label || head.display_label || head.label || head.id, { maxLength: 160 });
                const renameButton = button("Rename display label", () => manageProfile({ action: "rename", head_id: head.id, display_label: rename.value.trim() }, snapshot, context, identity, n)); renameButton.dataset.mutation = "";
                const archive = button(archived ? "Restore to selection" : "Archive from selection", () => manageProfile({ action: "archive", head_id: head.id, archived: !archived }, snapshot, context, identity, n)); archive.dataset.mutation = "";
                const download = h("a", "pruning-help", "Download existing profile bundle"); download.href = `${context.apiRoot}/data_ingestion/reference_profiles/${encodeURIComponent(head.id)}/export`; download.target = "_blank"; download.rel = "noopener";
                actionRow.append(rename, renameButton, archive, download); actions.append(actionRow); row.append(nameCell, training, association, actions); table.append(row);
            }
            const wrap = h("div", "pruning-table-wrap"); wrap.append(table); n.append(wrap);
            if (!heads().length) n.append(h("p", "pruning-help", "No trained profiles are available. Build a version with the existing ingestion trainer; encoder weights stay frozen."));
            const preferences = h("details", "pruning-disclosure"); preferences.open = true; preferences.append(h("summary", "", "Dataset preferred multi-head recipes"));
            const preferred = dataset.preferred_recipes || {};
            preferences.append(h("p", "pruning-help", `Whole-image preference: ${preferred.whole_image ? "saved" : "none"}; object preference: ${preferred.bbox_crop ? "saved" : "none"}. Preferences can contain multiple backbone-specific heads and fusion weights. Every referenced head must already be attached and compatible.`));
            const saveRecipe = role => {
                const draft = config();
                const components = role === "whole_image" ? draft.scene_components : (draft.object_components || state.capabilities?.inherited_recipe?.bbox?.components || state.capabilities?.inherited_recipe?.components || []);
                const resolved = components.map(v => { const head = heads().find(q => q.id === v.head_id); return { ...v, ...(head ? { head_sha256: head.head_sha256 } : {}) }; });
                manageProfile({ action: "prefer_recipe", role, recipe: { components: resolved, object_recipe: draft.object_recipe, adjustment: role === "bbox_crop" ? draft.object_adjustment : "per_component", ...(role === "bbox_crop" && draft.object_preprocessing ? { preprocessing: draft.object_preprocessing } : {}) } }, snapshot, context, identity, n);
            };
            const prefActions = h("div", "pruning-inline"), scene = button("Use draft whole-image recipe by default", () => saveRecipe("whole_image")), object = button("Use draft object recipe by default", () => saveRecipe("bbox_crop"));
            scene.dataset.mutation = object.dataset.mutation = ""; prefActions.append(scene, object); preferences.append(prefActions);
            if (Object.keys(preferred).length) preferences.append(h("pre", "pruning-help", JSON.stringify(preferred, null, 2))); n.append(preferences);
            const actions = h("div", "pruning-inline"); actions.append(button("Train a new whole-image version", () => { n.close(); adapters.openIngestionProfiles?.(); }), button("Import a profile with the existing tool", () => { n.close(); adapters.openIngestionProfiles?.(); document.getElementById("dataIngestionProfileUploadButton")?.click(); }), button("Refresh shared registry", () => openProfiles(n)));
            actions.append(button("Train / inspect crop SALAD in DQE", () => { n.close(); adapters.openDQE?.(); }));
            n.append(actions, h("p", "pruning-help", "Rename changes management metadata only, preserving checkpoint bytes and caches. Retraining creates a new immutable version. Imported bundles contain the head and checksum manifest, not backbone weights, datasets or vector caches. SAM3 mask features have no supported SALAD trainer here."));
            const status = h("p", "pruning-status"); status.dataset.profileStatus = ""; n.append(status); controls();
        }
        async function manageProfile(value, snapshot, context, identity, n) {
            if (state.busy || state.exporting || state.externalRefreshing || state.pending || !current(context)) return;
            const operation = ++state.operation, draft = config(); state.busy = true; controls();
            const status = n.querySelector('[data-profile-status]'); if (status) status.textContent = "Saving management metadata …";
            try {
                const answer = await mutation("profile", "/profiles", { expected_revision: snapshot.revision, source_identity: identity, ...value }, context);
                await capabilities(context); const fresh = await readProfiles(context, identity); applyConfig(draft);
                if (state.dialog === n && n.open) renderProfiles(n, fresh, context, identity);
                const currentStatus = n.querySelector('[data-profile-status]'); if (currentStatus) currentStatus.textContent = `Saved ${answer.receipt?.action || value.action}. Head bytes and historical training metadata are unchanged.`;
                message("Dataset profile management saved to the shared registry.");
            } catch (e) {
                if (current(context) && state.dialog === n) {
                    if (!state.pending && e.status === 409) { try { const fresh = await readProfiles(context, identity); renderProfiles(n, fresh, context, identity); } catch (_) { /* Keep the visible failure and outstanding authority. */ } }
                    const currentStatus = n.querySelector('[data-profile-status]'); if (currentStatus) { currentStatus.textContent = e.message; currentStatus.classList.add("pruning-error"); }
                }
                handleError(e, context);
            } finally { if (operation === state.operation) { state.busy = false; controls(); if (state.externalRefreshNeeded) refreshExternal(state.context); } }
        }
        async function exportVariant() {
            if (state.busy || state.exporting || state.externalRefreshing || state.reviewPending || state.pending || state.recoveryRequired || planningDirty() || !state.result) return;
            const operation = ++state.operation, context = state.context, jobId = state.jobId;
            const binding = { selection_revision: state.result.selection_revision, plan_revision: state.plan.plan_revision, expected_review_revision: state.review?.review_revision ?? state.plan.review_revision, removed_count: state.removed, split_policy: el.splitPolicy.value };
            state.exporting = true; controls(); el.exportStatus.textContent = "Saving annotations and validating the approved plan …";
            try {
                await adapters.checkpoint();
                if (!current(context)) throw new Error("Annotations changed since this analysis. Analyze again; compatible image features will be reused.");
                const job = await mutation("export", `/jobs/${encodeURIComponent(jobId)}/exports`, binding, context);
                state.exportId = job.export_id; state.exportBinding = binding; remember(); await pollExport(context);
            } catch (e) { if (current(context) && e.name !== "AbortError") el.exportStatus.textContent = e.message; }
            finally { if (operation === state.operation) { state.exporting = false; controls(); if (state.externalRefreshNeeded) refreshExternal(state.context); } }
        }
        async function pollExport(context) {
            const jobId = state.jobId, exportId = state.exportId, binding = state.exportBinding;
            while (current(context) && jobId === state.jobId && exportId === state.exportId) {
                const job = await request(`/jobs/${encodeURIComponent(jobId)}/exports/${encodeURIComponent(exportId)}`, {}, context);
                el.exportStatus.textContent = `${job.message || "Preparing approved variant"} · ${Math.round((job.progress || 0) * 100)}%`;
                if (binding && (job.selection_revision !== binding.selection_revision || (job.plan_revision !== undefined && job.plan_revision !== binding.plan_revision) || (job.review_revision !== undefined && job.review_revision !== binding.expected_review_revision))) {
                    invalidateExport(); el.exportStatus.textContent = "This variant belongs to an earlier review plan. Prepare the current approved plan again."; remember(); return;
                }
                if (job.status === "completed") { el.download.href = `${context.apiRoot}/dataset_pruning/jobs/${encodeURIComponent(jobId)}/exports/${encodeURIComponent(exportId)}/download`; el.download.hidden = false; remember(); return; }
                if (!["queued", "running"].includes(job.status)) throw new Error(job.message || `Export ${job.status}`);
                await pause(1000);
            }
        }
        async function restore(force = false) {
            if (!await verifyBackend()) return;
            const context = begin(adapters.context());
            if (state.busy || state.exporting) return;
            if (state.initialized && !force && !state.pending && !state.recoveryRequired) { renderSource(); return; }
            resetPlanning();
            const operation = ++state.operation; state.busy = true; controls();
            try {
                if (!context.apiRoot) { message("Connect a backend to configure pruning."); return; }
                const data = await capabilities(context);
                let saved = null; try { saved = JSON.parse(localStorage.getItem(storageKey(context)) || localStorage.getItem(`tator.pruning.v1:${signature(context)}`) || "null"); if (saved?.pending && !saved.pending.context_signature) saved.pending.context_signature = signature(context); } catch (_) { message("Saved browser recovery could not be read. Server review receipts remain authoritative.", true); }
                state.savedLab = saved?.recipeLab || null;
                if (saved?.pending) { state.pending = saved.pending; remember(); }
                state.pending = unresolved(context)[0] || null;
                state.jobId = (state.pending?.kind !== "profile" && state.pending?.annotation_scope === sourceKey(context) ? state.pending?.jobId : "") || saved?.jobId || ""; state.removed = saved?.removed || 0; state.lastBatch = saved?.lastBatch || "";
                state.exportId = saved?.exportId || ""; state.exportBinding = saved?.exportBinding || null;
                state.planning.view = saved?.reviewView === "family" ? "family" : "impact";
                el.filter.value = saved?.filter || "all"; el.search.value = saved?.query || ""; el.sort.value = saved?.sort || "stable";
                state.noteDrafts = new Map(saved?.noteDrafts || []); state.familyDrafts = new Map((saved?.familyDrafts || []).map(([id, value]) => [id, { ...value, preview: null, serial: 0 }]));
                if (sourceKey(context)) await readProfiles(context);
                const preferred = state.profiles?.dataset?.preferred_recipes || {};
                const draft = saved?.draft || { ...(data.default_config || {}), ...(preferred.whole_image ? { scene_components: preferred.whole_image.components } : {}), ...(preferred.bbox_crop ? { object_recipe: preferred.bbox_crop.object_recipe || "inherit_dqe", object_components: preferred.bbox_crop.components, object_adjustment: preferred.bbox_crop.adjustment || "inherit", ...(preferred.bbox_crop.preprocessing ? { object_preprocessing: preferred.bbox_crop.preprocessing } : {}) } : {}) };
                applyConfig(draft); state.initialized = true;
                if (state.pending) {
                    const kind = state.pending.kind, pendingValue = state.pending.value, answer = await mutation(kind, state.pending.path, pendingValue, context, true);
                    if (kind === "analysis") state.jobId = answer.job_id;
                    if (kind === "export") { state.exportId = answer.export_id; state.exportBinding = { ...pendingValue, ...(saved?.exportBinding || {}) }; }
                    if (kind === "review") { state.review = answer.snapshot || null; if (pendingValue.action === "bulk_commit") state.lastBatch = answer.receipt?.batch_id || answer.snapshot?.last_batch_id || state.lastBatch; if (pendingValue.action === "bulk_undo") state.lastBatch = lastBatch(answer.snapshot); invalidateExport(); }
                    if (kind === "profile") { await capabilities(context); await readProfiles(context); applyConfig(draft); }
                    if (kind === "calibration") message("Recovered the saved recipe calibration. Load comparison examples to inspect its profile; removals were not approved.");
                    remember();
                }
                if (state.jobId) { await poll(context, true); if (state.exportId) { state.exporting = true; await pollExport(context); } }
                else message(context.annotationSource?.id ? "Choose the shared feature recipe and analyze when ready. No analysis starts on tab entry." : "Open or attach a dataset to begin. Recipe setup is available here without opening DQE first.");
            } catch (e) { if (current(context) && e.name !== "AbortError") { state.recoveryRequired = !!state.jobId || !!state.pending; message(e.message, true); } }
            finally { if (operation === state.operation) { state.busy = false; state.exporting = false; controls(); if (state.externalRefreshNeeded) refreshExternal(state.context); } }
        }
        function chooseObjectRecipe() {
            state.objectCustom = false; state.preparationCustom = false;
            const recipe = el.objectRecipe.value;
            el.objectAdjustment.value = recipe === "demo_raw_dino_sam" ? "raw" : "inherit";
            objectControls({ object_recipe: recipe });
            if (recipe === "demo_raw_dino_sam") { el.objectAdjustment.value = "raw"; el.rulePreset.value = "demo"; applyRulePreset(); }
            else { el.rulePreset.value = "draft"; draftChanged(); }
        }
        function applyRulePreset() {
            if (el.rulePreset.value === "demo") { el.objectRecipe.value = "demo_raw_dino_sam"; el.objectAdjustment.value = "raw"; state.objectCustom = false; state.preparationCustom = false; objectControls({ object_recipe: "demo_raw_dino_sam" });
                const demoHead = heads().find(v => v.id === "WALDO3" && v.head_sha256 === "be870ed5e5704fd00594721cb17700d240e1eb212507788a1ce6f94611819ac5");
                if (demoHead) sceneControls([{ encoder: "local_salad", encoder_type: demoHead.encoder_type, model: demoHead.encoder_model, head_id: demoHead.id, pooling: demoHead.pooling || "pooler", weight: 1, adjustment: "raw" }]);
                el.mode.value = "compound"; el.scaleProtection.value = "occupancy_legacy"; el.sceneRequirement.value = "required"; el.objectWeight.value = .9; el.sceneThreshold.value = .92; el.objectThreshold.value = .80; el.indexThreshold.value = .87; el.rarityScale.value = .60; }
            if (el.rulePreset.value === "strict") { el.sceneThreshold.value = .90; el.objectThreshold.value = .90; el.indexThreshold.value = .94; el.rarityScale.value = .80; }
            draftChanged();
        }
        el.run.addEventListener("click", run); el.reconnect.addEventListener("click", () => restore(true));
        el.openDQE.addEventListener("click", () => adapters.openDQE?.());
        el.cancel.addEventListener("click", () => { const context = state.context, jobId = state.jobId; request(`/jobs/${encodeURIComponent(jobId)}/cancel`, post({}), context).catch(e => handleError(e, context)); });
        el.objectRecipe.addEventListener("change", chooseObjectRecipe); el.rulePreset.addEventListener("change", applyRulePreset);
        el.objectAdjustment.addEventListener("change", () => { const policy = el.objectAdjustment.value; for (const row of state.objectRows) row.strength.value = policy === "raw" ? 0 : policy === "remove_all" ? 1 : policy === "remove" ? (row.id === "cradio" ? 0 : 1) : row.inheritedStrength; draftChanged(); });
        for (const k of ["mode", "objectAdjustment", "objectWeight", "objectThreshold", "sceneThreshold", "indexThreshold", "sceneRequirement", "negativeThreshold", "rarityScale", "neighborK", "scaleProtection"]) el[k].addEventListener("input", () => { el.rulePreset.value = "draft"; draftChanged(); });
        el.objectAdvanced.addEventListener("change", event => { if (state.objectRows.some(v => [v.enabled, v.weight, v.pooling, v.head, v.model, v.strength].includes(event.target))) state.objectCustom = true; if (Object.values(state.preparation).includes(event.target)) state.preparationCustom = true; el.rulePreset.value = "draft"; draftChanged(); });
        for (const k of ["slider", "count"]) el[k].addEventListener("input", () => budget(el[k].value));
        el.percent.addEventListener("input", () => { if (state.result) budget(Math.floor(Number(el.percent.value) * state.result.image_count / 100)); });
        el.policyStrict.addEventListener("click", () => editSettings({ policy: "strict_duplicates" }));
        el.policyBudget.addEventListener("click", () => editSettings({ policy: "budgeted_coverage" }));
        el.planObjectWeight.addEventListener("input", () => editSettings({ object_weight: Number(el.planObjectWeight.value) }));
        for (const k of ["lossGroupRange", "lossGroupCap"]) el[k].addEventListener("input", () => editSettings({ max_affected_appearance_groups: Number(el[k].value) }));
        for (const k of ["lossPercentRange", "lossPercentCap"]) el[k].addEventListener("input", () => editSettings({ max_lost_appearance_weight_fraction: Number(el[k].value) / 100 }));
        el.applyPlan.addEventListener("click", applySettings);
        el.discardPlan.addEventListener("click", () => discardSettings().catch(e => handleError(e)));
        el.recalculatePlan.addEventListener("click", () => editSettings({}, true));
        el.impactView.addEventListener("click", () => setReviewView("impact").catch(e => handleError(e)));
        el.familyView.addEventListener("click", () => setReviewView("family").catch(e => handleError(e)));
        let dragging = false;
        const pointerBudget = event => { const transform = el.chart.getScreenCTM(); if (!transform || !state.result) return; const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(transform.inverse()); budget((point.x - 58) / 580 * (Number(el.chart.dataset.maximumImages) || chartExtent() * state.result.image_count / 100)); };
        el.chart.addEventListener("pointerdown", event => { if (state.busy || state.exporting || state.recoveryRequired || state.pending || el.chart.dataset.planSource === "outdated") return; dragging = true; el.chart.setPointerCapture(event.pointerId); pointerBudget(event); });
        el.chart.addEventListener("pointermove", event => { if (dragging) pointerBudget(event); }); for (const k of ["pointerup", "pointercancel"]) el.chart.addEventListener(k, () => { dragging = false; });
        el.bulkPreview.addEventListener("click", bulkPreview); el.bulkUndo.addEventListener("click", () => reviewAction({ action: "bulk_undo", batch_id: state.lastBatch }));
        for (const k of ["bulkCount", "bulkScope"]) el[k].addEventListener("change", () => invalidatePreview());
        for (const k of ["search", "filter", "sort"]) el[k].addEventListener(k === "search" ? "input" : "change", () => { if (!state.result || state.busy || state.pending) return; state.serial++; invalidatePreview(); clearTimeout(state.familyTimer); state.familyTimer = setTimeout(() => families(state.context, true).catch(e => handleError(e)), 180); });
        const moreReviews = () => (state.planning.view === "impact" ? impactCards(state.context) : families(state.context)).catch(e => handleError(e));
        el.loadMore.addEventListener("click", moreReviews);
        const observer = typeof IntersectionObserver === "function" ? new IntersectionObserver(entries => { if (entries.some(v => v.isIntersecting) && active() && (state.planning.view === "impact" ? state.planning.next !== null && !state.planning.loading : state.next !== null) && !state.reviewPending && !state.busy && !state.pending) moreReviews(); }, { rootMargin: "600px" }) : null; observer?.observe(el.sentinel);
        el.export.addEventListener("click", exportVariant); el.splitPolicy.addEventListener("change", () => { invalidateExport(); remember(); });
        const externalClick = event => {
            const target = event.target.closest?.("[data-pruning-open-profiles],[data-pruning-open-tab]"); if (!target) return;
            if (target.hasAttribute("data-pruning-open-profiles")) openProfiles(); else document.getElementById("tabDatasetPruningButton")?.click();
        };
        document.addEventListener("click", externalClick);
        const watcher = setInterval(() => {
            if (!state.context || state.checkpointing || current(state.context)) return;
            begin(adapters.context()); message("Dataset or annotations changed. The old review is suspended; analyze the current source to reuse compatible pixel features.");
            if (active()) restore().catch(e => handleError(e));
        }, 1000);
        const onHide = () => { remember(); state.controller?.abort(); };
        root.addEventListener("pagehide", onHide);
        const onStorage = event => {
            if (!state.context || !event.key?.startsWith(outboxPrefix(state.context))) return;
            try {
                const hadPending = !!state.pending || state.recoveryRequired;
                state.pending = unresolved(state.context)[0] || null; state.recoveryRequired = !!state.pending;
                if (state.pending) message("Another tab has an unresolved operation for this dataset. Reconnect to recover its exact receipt.", true);
                else if (hadPending) { state.reviewPending = true; refreshExternal(state.context); }
                controls();
            } catch (error) { state.recoveryRequired = true; controls(); message(error.message, true); }
        };
        async function refreshExternal(context) {
            if (!context || !current(context)) return;
            if (state.busy || state.exporting || state.externalRefreshing || state.pending) { state.externalRefreshNeeded = true; return; }
            state.externalRefreshNeeded = false; state.externalRefreshing = true; state.reviewPending = true; controls();
            try {
                resetPlanning();
                if (state.result && state.jobId) await refreshReview(context);
                await capabilities(context); if (sourceKey(context)) await readProfiles(context);
                if (current(context)) { state.recoveryRequired = false; message("Recovered the committed operation from another tab. Current keepers, approvals and profiles are refreshed from the server."); }
            } catch (error) { if (current(context)) { state.recoveryRequired = true; message(error.message, true); } }
            finally { if (current(context)) { state.externalRefreshing = false; state.reviewPending = false; controls(); if (state.externalRefreshNeeded) refreshExternal(context); } }
        }
        root.addEventListener("storage", onStorage);
        recipeLab = root.TatorPruningRecipeLab?.create({ host, state, contextIsCurrent: current, request, mutation, applyConfig, config, remember, controls, thumbnail, assetURL, className, notify: message, onSettled: () => { if (state.externalRefreshNeeded) refreshExternal(state.context); } });
        const instance = { activate: () => restore(), refresh: () => restore(true), openProfiles, destroy: () => { cancelSettingsPreview(); for (const value of state.familyDrafts.values()) clearTimeout(value.timer); clearInterval(watcher); clearTimeout(state.timer); clearTimeout(state.familyTimer); observer?.disconnect(); state.controller?.abort(); document.removeEventListener("click", externalClick); root.removeEventListener("pagehide", onHide); root.removeEventListener("storage", onStorage); state.dialog?.remove(); delete host.pruningInstance; } };
        host.pruningInstance = instance; el.sceneComponents.append(h("p", "pruning-help", "Connect a backend to load available whole-image backbones and compatible SALAD profiles.")); el.objectAdvanced.append(h("p", "pruning-help", "Component controls appear after the backend recipe capabilities are loaded.")); renderSource(); controls(); return instance;
    }
    root.TatorDatasetPruning = Object.freeze({ create });
}(typeof globalThis !== "undefined" ? globalThis : this));
