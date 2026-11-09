/* Administrator entry to snapshot-bound analysis and retained native results. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const node = (tag, text) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; return element; };
    const count = (value, one, many = `${one}s`) => `${Number(value || 0).toLocaleString()} ${value === 1 ? one : many}`;
    // Opened from the administrator navigation: "Server jobs" or one named tool.
    const dialog = node("dialog"); dialog.id = "hostedAnalysisDialog";
    const notice = node("p"); notice.setAttribute("role", "status");
    const controls = node("section"), history = node("section"), content = node("section");
    const close = node("button", "Close"); close.onclick = () => dialog.close();
    const title = node("h2", "Server jobs");
    dialog.append(title, node("p", "Every run uses a fixed copy of the confirmed image list and annotation versions. Incoming work is excluded. Heavy compute pauses annotation assistance; manual editing and saving continue."), notice, controls, history, content, close);
    document.body.append(dialog);
    let epoch = 0, datasetId = "", currentJob = null, timer = null, activeRequest = null, cancel = null, selectedOperation = null;
    const operationNames = {dqe: "Data Quality Explorer", pruning: "Dataset pruning", ingestion: "Data ingestion", clip_training: "Train class predictor", yolo_training: "Train YOLO", sam3_training: "Train SAM3", rfdetr_training: "Train RF-DETR", qwen_training: "Train Qwen", salad_training: "Train SALAD descriptor", caption_export: "Export captions for training"};
    const consumerNames = {...operationNames, statistics: "Dataset statistics", dqe_review: "Qwen visual review"};
    // Trainers whose upstream licence must be accepted for each run, as in the workbench.
    const licences = {yolo_training: "I have read and accept the Ultralytics YOLO licence terms", rfdetr_training: "I have read and accept the RF-DETR licence terms"};
    // Launcher order: the quick, read-only tasks first; exports last.
    const taskOrder = ["statistics", "dqe", "pruning", "ingestion", "clip_training", "yolo_training", "rfdetr_training", "sam3_training", "qwen_training", "salad_training", "caption_export"];
    const trainingTasks = new Set(["clip_training", "yolo_training", "sam3_training", "rfdetr_training", "qwen_training"]);
    const stateNames = {admitting: "Preparing inputs", queued: "Waiting for its turn", running: "Running", completed: "Complete", complete: "Complete", captured: "Started", failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted", stale: "Out of date", unavailable: "Unavailable"};
    const computeErrors = {native_engine_failed: "The analysis engine stopped with an error; its log is kept on the server.", worker_timeout: "The task ran past its time limit and was stopped.",
        worker_operation_failed: "The task failed on the server; its log is kept on the server.", worker_failed: "The task failed on the server; its log is kept on the server.", worker_disconnected: "The compute worker stopped unexpectedly; its log is kept on the server.", worker_protocol_invalid: "The compute worker sent an unreadable reply; its log is kept on the server.", worker_termination_failed: "The compute worker could not be stopped cleanly; an administrator should check the server.",
        runtime_cancelled: "The task was cancelled.", request_cancelled: "The task was cancelled.", runtime_closed: "The compute service shut down before the task finished. Start it again.", runtime_queue_full: "Too many tasks are waiting. Try again when the current task finishes.", runtime_heavy_window_busy: "Another administrator task is running. Try again when it finishes.", runtime_configuration_changed: "The server's model configuration changed. Start the task again.", runtime_authority_changed: "Your account changed while the task ran. Sign in again and restart it.", runtime_owner_limit: "You already have the maximum number of running tasks.", runtime_payload_budget_exceeded: "The task's input is too large for this server's limits.",
        heavy_disk_insufficient: "The server does not have enough free scratch disk space for this task.", heavy_tool_not_configured: "This task is not configured on this server.", heavy_job_interrupted: "The task was interrupted, for example by a server restart. Start it again.", native_analysis_interrupted: "The task was interrupted, for example by a server restart. Start it again.", heavy_job_authority_revoked: "Your administrator access changed while the task ran.",
        native_model_not_verified: "A model file failed verification. An administrator must reinstall it.", native_model_environment_invalid: "A model's runtime environment is not set up correctly on this server.", analysis_validation_split_required: "Training needs training and validation images; add images or capture groups so some can be held out.",
        analysis_snapshot_stale: "The confirmed images changed since this run was prepared. Start a new run.", analysis_snapshot_identity_changed: "The confirmed images changed while the task was preparing. Start it again.", analysis_media_changed: "An image file changed while the task was preparing. Start it again.", analysis_manifest_changed: "The confirmed images changed while the task was preparing. Start it again.", analysis_materialization_changed: "The task's prepared images changed. Start it again.", native_annotation_identity_changed: "Annotations changed while the task was running. Start it again.", native_cache_input_changed: "A cached input changed while the task was running. Start it again.",
        native_yolo_terms_required: "YOLO training needs the Ultralytics licence accepted: tick the licence box when you start the task.", native_rfdetr_terms_required: "RF-DETR training needs the RF-DETR licence accepted: tick the licence box when you start the task.",
        native_gpu_unavailable: "The task needs a GPU, but none is available to it on the server. Ask the operator to check the GPU or the task's device setting.", native_classifier_classes_missing: "The class predictor needs confirmed objects of at least two classes in its training images.", native_dqe_scope_required: "This Data Quality Explorer task needs a class, or all classes, in its server configuration.", native_model_unavailable: "A model this task needs is not installed for server jobs. Ask the operator to add it to the task's verified model files.",
        analysis_result_unavailable: "This result is no longer available on the server.", dqe_result_unavailable: "This result is no longer available on the server.", analysis_manifest_too_large: "This dataset is too large for this task's limits.", ingestion_salad_head_not_attached: "Attach a whole-image SALAD descriptor in Analysis presets first.", ingestion_candidate_not_unassigned_incoming: "Some selected images are no longer Not assigned in Incoming. Select them again."};
    const computeError = code => !code ? "" : computeErrors[code] || app.errorText?.(code) || String(code).replaceAll("_", " ");
    const error = (problem) => { notice.textContent = problem.message; notice.dataset.error = "true"; };
    function act(parent, title, callback) {
        const ticket = epoch; const b = node("button", title); b.type = "button";
        b.onclick = async () => { if (!valid(ticket)) return; b.disabled = true; try { await callback(); } catch (problem) { if (valid(ticket) && !problem.stale) error(problem); } finally { b.disabled = false; } };
        parent.append(b); return b;
    }
    function field(parent, label, input) { const wrap = node("label"); input.id = input.id || `analysis-field-${crypto.randomUUID()}`; wrap.htmlFor = input.id; wrap.append(node("span", label), input); input.setAttribute("aria-label", label); parent.append(wrap); return input; }
    function select(values) { const element = node("select"); for (const [value, label] of values) element.add(new Option(label, value)); return element; }
    const api = (path, body) => app.api(path, body === undefined ? {} : { method: "POST", body });
    const valid = (ticket) => ticket === epoch && dialog.open && datasetId === app.datasetId && app.session?.user.role === "admin";
    async function refreshHistory(ticket, offset = 0) {
        const result = await api(`/analysis/jobs?dataset_id=${encodeURIComponent(datasetId)}&offset=${offset}&limit=50`);
        if (!valid(ticket)) return;
        history.querySelector('[data-older-analyses]')?.remove();
        if (!offset) history.replaceChildren(node("h3", "Saved analyses"));
        const allRows = result.jobs || [], rows = allRows.filter(row => !selectedOperation || row.consumer === selectedOperation);
        if (!rows.length && !offset) history.append(node("p", selectedOperation ? "No saved results for this tool in the most recent jobs." : "No saved results for this dataset yet."));
        for (const row of rows) {
            const item = node("div"); item.className = "hosted-analysis-job";
            item.append(node("strong", `${consumerNames[row.consumer] || row.consumer.replaceAll("_", " ")} · ${stateNames[row.state] || row.state}`), node("small", `${Number.isFinite(row.confirmed_count) ? count(row.confirmed_count, "confirmed image") : "—"} · ${new Date(row.created_at * 1000).toLocaleString()}${row.stale && row.state === "complete" ? " · earlier dataset version" : ""}${row.error ? ` · ${computeError(row.error)}` : ""}`));
            if (row.state === "complete") act(item, "Open results", () => showResult(row.id || row.job_id, ticket));
            else if (row.state === "captured" && row.runtime_request_id) act(item, "Follow this task", () => track(row.runtime_request_id, ticket));
            history.append(item);
        }
        if (allRows.length === 50) act(history, "Show older analyses", () => refreshHistory(ticket, offset + 50)).dataset.olderAnalyses = "";
    }
    async function watch(requestId, ticket) {
        const state = await api(`/tools/requests/${encodeURIComponent(requestId)}`);
        if (!valid(ticket) || activeRequest !== requestId) return;
        const progress = state.state === "admitting" && state.progress?.total ? ` · ${state.progress.done.toLocaleString()} of ${state.progress.total.toLocaleString()} images` : "";
        notice.textContent = state.state === "admitting"
            ? `Preparing inputs: ${(state.phase || "snapshot").replaceAll("_", " ")}${progress}. You can cancel until the run starts.`
            : `${stateNames[state.state] || state.state.replaceAll("_", " ")}${state.error ? ": " + computeError(state.error) : ""}`;
        notice.dataset.error = String(Boolean(state.error) || ["failed", "interrupted", "unavailable"].includes(state.state));
        if (["completed", "failed", "cancelled", "interrupted", "stale", "unavailable"].includes(state.state)) {
            activeRequest = null; cancel?.remove(); cancel = null;
            await refreshHistory(ticket);
            if (state.state === "completed" && state.analysis_job_id) await showResult(state.analysis_job_id, ticket);
        } else timer = setTimeout(() => watch(requestId, ticket).catch(e => { if (valid(ticket) && !e.stale) error(e); }), 1000);
    }
    async function track(requestId, ticket) {
        clearTimeout(timer); activeRequest = requestId; cancel?.remove();
        cancel = act(controls, "Cancel this compute task", async () => {
            if (activeRequest !== requestId) return;
            await api(`/tools/requests/${encodeURIComponent(requestId)}/cancel`, {});
            if (valid(ticket)) notice.textContent = "Cancellation requested. Waiting for assistance to be restored.";
        });
        await watch(requestId, ticket);
    }
    function parameterControl(parent, key, value, operation) {
        const labels = { quality_recipe: "Object feature recipe", epochs: "Training epochs", batch_size: "Batch size", imgsz: "Training image size", device: "Device", model: "Starting model", object_weight: "Object contribution", object_threshold: "Minimum object similarity", scene_threshold: "Minimum whole-image similarity", index_threshold: "Minimum combined similarity", neighbor_k: "Candidate neighbors per branch", qa_mix: "Question mix", answer_format: "Answer style", lr: "Learning rate", learning_rate: "Learning rate", patience: "Early-stopping patience", workers: "Data loader workers", seed: "Random seed" };
        let input;
        if (key === "quality_recipe") input = select([["thorough_quality_v1", "DQE Thorough · DINOv3 + SAM3 + crop SALAD + C-RADIO"], ["precise_compact_v1", "DQE Compact · DINOv3 + SAM3 + crop SALAD"], ["fast_map_v1", "DQE Fast · DINOv3"]]);
        else if (typeof value === "boolean") { input = node("input"); input.type = "checkbox"; input.checked = value; }
        else if (value && typeof value === "object") { input = node("textarea"); input.rows = 3; input.value = JSON.stringify(value, null, 2); }
        else { input = node("input"); input.type = typeof value === "number" ? "number" : "text"; input.value = value ?? ""; if (input.type === "number") input.step = "any"; }
        if (key === "quality_recipe") input.value = value || "thorough_quality_v1";
        const captionLabels = {accept_tos: licences[operation] || "I accept the model licence terms", include_caption0_in_training: "Include primary captions", include_generated_qa_in_training: "Include generated questions and answers", include_deterministic_metadata_qa: "Include questions derived from labels", instruction_qa_restrict_speculative_language: "Reject speculative answers", require_ready_instruction_export: "Require readiness checks to pass before creating a training bundle"};
        const text = labels[key] || captionLabels[key] || `${key.charAt(0).toUpperCase()}${key.slice(1).replaceAll("_", " ")}`;
        if (input.type === "checkbox") {
            // A checkbox sits in front of its label, on one line.
            const wrap = node("label"); wrap.className = "hosted-checkbox"; input.id = `analysis-field-${crypto.randomUUID()}`; wrap.htmlFor = input.id;
            wrap.append(input, node("span", text)); parent.append(wrap);
        } else field(parent, text, input);
        return () => input.type === "checkbox" ? input.checked : typeof value === "number" ? Number(input.value) : value && typeof value === "object" ? JSON.parse(input.value) : input.value;
    }
    async function open(preferredOperation) {
        if (app.session?.user.role !== "admin") return;
        selectedOperation = operationNames[preferredOperation] ? preferredOperation : null;
        const name = app.datasetName;
        title.textContent = `${operationNames[selectedOperation] || "Server jobs"}${name ? ` · ${name}` : ""}`;
        disposeGraph?.(); disposeGraph = null;
        clearTimeout(timer); currentJob = null; activeRequest = null; cancel?.remove(); cancel = null; const ticket = ++epoch; datasetId = app.datasetId;
        notice.textContent = ""; content.replaceChildren(); controls.replaceChildren(); history.replaceChildren(); dialog.showModal();
        if (!datasetId) { notice.textContent = "Choose a dataset first."; return; }
        const catalog = await api(`/tools/catalog?dataset_id=${encodeURIComponent(datasetId)}`);
        if (!valid(ticket)) return;
        const rank = value => { const index = taskOrder.indexOf(value.operation); return index < 0 ? taskOrder.length : index; };
        const options = Object.entries(catalog.heavy_job_options || {}).filter(([, value]) => value.operation !== "dqe_review" && (!selectedOperation || value.operation === selectedOperation))
            .sort(([, a], [, b]) => rank(a) - rank(b));
        const profiles = options.some(([, value]) => value.operation === "ingestion") ? await api(`/datasets/${datasetId}/salad-profiles`) : null;
        if (!valid(ticket)) return;
        const candidates = app.selectedImages.filter(image => image.dataset_id === datasetId && image.state === "unassigned");
        if (!options.length) controls.append(node("p", "This server has no configured runtime for this task yet. Saved results remain available below."));
        else {
            const recipe = field(controls, "Compute task", select(options.map(([id, value]) => [id, value.label || value.operation])));
            const preferred = options.find(([, value]) => value.operation === preferredOperation);
            if (preferred) recipe.value = preferred[0];
            const parameters = node("div"); parameters.className = "hosted-analysis-parameters"; controls.append(parameters);
            let readers = {};
            function render() {
                parameters.replaceChildren(); readers = {};
                const definition = catalog.heavy_job_options[recipe.value];
                for (const key of definition.allowed_parameters || []) {
                    if (definition.operation === "ingestion" && ["candidate_image_ids", "salad_head_id"].includes(key)) continue;
                    const value = key === "accept_tos" ? Boolean(definition.defaults?.[key]) : definition.defaults?.[key];
                    if (value !== undefined || key === "quality_recipe") readers[key] = parameterControl(parameters, key, value, definition.operation);
                }
                if (licences[definition.operation] && !(definition.allowed_parameters || []).includes("accept_tos")) {
                    parameters.append(node("p", "This trainer runs only after its licence is accepted, and this task's server configuration has no licence setting, so the run would be refused. Ask the operator to allow accept_tos for this task."));
                }
                if (definition.operation === "ingestion") {
                    parameters.append(node("p", `${count(candidates.length, "selected Incoming image")} will be compared with confirmed images. To change this set, close this dialog and select images in Incoming. Results are suggestions; images still need annotation and approval.`));
                    const heads = (profiles?.heads || []).filter(head => !head.archived && head.representation_scope === "whole_image" && profiles.dataset.head_ids.includes(head.id));
                    const selectedHead = field(parameters, "Trained whole-image SALAD descriptor", select(heads.map(head => [head.id, head.display_label || head.label || head.id])));
                    readers.salad_head_id = () => { if (!selectedHead.value) throw new Error("Attach a whole-image SALAD descriptor in Analysis presets, or train one first."); return selectedHead.value; };
                    readers.candidate_image_ids = () => { if (!candidates.length) throw new Error("Select Not assigned images in Incoming before starting this comparison."); return candidates.map(image => image.id); };
                }
                if (trainingTasks.has(definition.operation)) parameters.append(node("p", "Splits set on the dataset are used. Unsplit images are split for this run only (about 10% validation, capture groups kept together); the dataset is not changed."));
                if (definition.operation === "salad_training") parameters.append(node("p", "Train a whole-image descriptor on confirmed images. After training, open Analysis presets to register and attach the saved descriptor."));
                if (definition.operation === "caption_export") parameters.append(node("p", "Build the existing Tator caption and question-answer training archive from confirmed annotations. The original readiness checks still apply. Uncheck the readiness requirement only to download a diagnostic bundle; this does not make its contents ready for training."));
                if (!Object.keys(readers).length) parameters.append(node("p", "This task uses its server-configured recipe. The saved result records the resolved settings."));
            }
            recipe.onchange = render; render();
            act(controls, "Start task", async () => {
                const parameters = Object.fromEntries(Object.entries(readers).map(([key, read]) => [key, read()]));
                const definition = catalog.heavy_job_options[recipe.value];
                if (licences[definition.operation] && readers.accept_tos && !parameters.accept_tos) throw new Error("Tick the licence box to start this training.");
                const dataset = await api(`/datasets/${encodeURIComponent(datasetId)}`);
                if (!valid(ticket)) return;
                // Every task except the Incoming comparison works on the confirmed images.
                if (definition.operation !== "ingestion" && !dataset.counts?.confirmed) throw new Error(`${dataset.name} has no confirmed images yet. Confirm images in the dataset workspace before starting this task.`);
                const scope = definition.operation === "ingestion" ? `${count(candidates.length, "selected Incoming image")}, compared with ${count(dataset.counts?.confirmed || 0, "confirmed image")}`
                    : `all ${count(dataset.counts?.confirmed || 0, "confirmed image")}`;
                if (!window.confirm(`Start "${definition.label || definition.operation}" on ${scope} of ${dataset.name}?\n\nLarge datasets can take hours. Annotation assistance pauses for everyone while it runs, and the server needs scratch disk space roughly the size of these images. You can cancel it from this dialog.`)) return;
                const body = { dataset_id: datasetId, job: recipe.value, operation_id: crypto.randomUUID(), parameters };
                // Preparing a large dataset can take minutes. An exact replay of the
                // same command returns its ID and progress at once, so it can be
                // followed and cancelled while preparation continues.
                const first = api("/tools/heavy", body); first.catch(() => {});
                const early = await Promise.race([first.then(value => ({value}), problem => ({problem})), new Promise(resolve => setTimeout(() => resolve(null), 1500))]);
                if (early?.problem?.status) throw early.problem;
                const result = early?.value || await api("/tools/heavy", body);
                if (!valid(ticket)) return;
                notice.textContent = "Preparing inputs. You can cancel until the run starts.";
                await refreshHistory(ticket); await track(result.id, ticket);
            });
        }
        await refreshHistory(ticket);
    }
    async function showResult(jobId, ticket) {
        await renderResult(jobId, ticket);
        // The result opens below the saved list; bring it into view once drawn.
        if (valid(ticket) && currentJob === jobId) content.scrollIntoView({block: "start"});
    }
    async function renderResult(jobId, ticket) {
        disposeGraph?.(); disposeGraph = null; currentJob = jobId;
        const job = await api(`/analysis/jobs/${jobId}`);
        if (!valid(ticket) || currentJob !== jobId) return;
        content.replaceChildren(node("h3", job.consumer === "dqe" ? "Data Quality Explorer" : "Analysis result"));
        content.append(node("p", job.stale ? "Confirmed data has changed since this run. These results remain available as history; create a new run before applying changes." : "This result describes the current confirmed dataset."));
        if (job.consumer === "pruning") {
            const a = node("a", "Open Dataset pruning →"); a.href = `/pruning?job=${encodeURIComponent(jobId)}`; a.className = "hosted-download"; content.append(a); return;
        }
        if (job.consumer === "dqe") { await showGraph(jobId, ticket); return; }
        const response = await api(`/analysis/jobs/${jobId}/result`);
        if (!valid(ticket) || currentJob !== jobId) return;
        const result = response.result?.engine_result || response.result?.result || response.result || response;
        if (job.consumer === "statistics") {
            content.append(node("p", count(result.image_count ?? job.confirmed_count, "confirmed image")));
            if (result.splits) {
                const names = [["train", "training"], ["val", "validation"], ["test", "test"], ["unsplit", "not split"]];
                content.append(node("p", `Splits set on the dataset: ${names.map(([key, name]) => `${Number(result.splits[key] || 0).toLocaleString()} ${name}`).join(" · ")}`));
            }
            content.append(node("h4", "Objects per class"));
            const list = node("dl");
            for (const [label, count] of Object.entries(result.object_counts || {})) list.append(node("dt", label), node("dd", String(count)));
            content.append(list);
        } else if (job.consumer === "caption_export") {
            const summary = result.summary || {};
            content.append(node("strong", result.blocked_reason ? "Training readiness needs attention" : "Training readiness checks passed"));
            content.append(node("p", `${count(result.image_count, "confirmed image")} · ${count(summary.instruction_training_row_count || 0, "training example")} · ${count(summary.manual_review_required_count || 0, "row")} to review.`));
            if (result.blocked_reason) content.append(node("p", `The original export check reported: ${result.blocked_reason.replaceAll("_", " ")}. The readiness report contains the affected examples and checks.`));
            if (result.diagnostic_export) content.append(node("strong", "Diagnostic bundle — readiness checks have not passed."));
            for (const [path, title] of [[result.bundle_path, result.diagnostic_export ? "Download diagnostic bundle" : "Download caption training bundle"], [result.report_path, "Download readiness report"], [result.export_path, "Download captions and questions"]]) {
                if (!path) continue;
                const link = node("a", title); link.className = "hosted-download"; link.href = `/api/hosted/analysis/jobs/${encodeURIComponent(jobId)}/artifact?path=${encodeURIComponent(path)}`; link.download = ""; content.append(link);
            }
        } else if (job.consumer === "ingestion") {
            const measured = result.result || result, items = measured.items || [];
            content.append(node("p", `${items.filter(item => item.keep).length} of ${items.length} incoming images selected by the saved diversity recipe. Reference novelty measures difference from confirmed images; diversity measures variety within the incoming set. These images remain in Incoming.`));
            if (measured.distribution?.points?.length) {
                const Plotly = await window.TatorHostedDQE.loadPlotly();
                if (!valid(ticket) || currentJob !== jobId) return;
                const layout = node("div"), graph = node("div"), selected = node("aside"); layout.className = "hosted-analysis-graph";
                graph.setAttribute("aria-label", "Incoming and confirmed image features");
                selected.append(node("p", "Click a dot to see its image. Gray dots are confirmed images; colored dots are Incoming. Nearby dots have similar whole-image features in this two-dimensional view."));
                layout.append(graph, selected); content.append(layout);
                const points = measured.distribution.points.filter(point => point.hosted_identity && point.projection?.length === 2 && point.projection.every(Number.isFinite));
                const css = getComputedStyle(document.documentElement);
                const traces = [["reference", "Confirmed images", css.getPropertyValue("--muted") || "#8997a8"], ["candidate", "Incoming images", "#d47433"]].map(([kind, name, color]) => {
                    const rows = points.filter(point => point.kind === kind);
                    return {type: "scattergl", mode: "markers", name, x: rows.map(point => point.projection[0]), y: rows.map(point => point.projection[1]), customdata: rows,
                        text: rows.map(point => point.filename), hovertemplate: "%{text}<extra></extra>", marker: {size: kind === "candidate" ? 10 : 5, opacity: .75, color}};
                });
                await Plotly.newPlot(graph, traces, {height: 380, margin: {l: 40, r: 10, t: 10, b: 45}, paper_bgcolor: css.getPropertyValue("--surface"), plot_bgcolor: css.getPropertyValue("--surface"), font: {color: css.getPropertyValue("--text")}, legend: {orientation: "h"}, xaxis: {title: "Feature direction 1"}, yaxis: {title: "Feature direction 2"}}, {responsive: true, displaylogo: false, modeBarButtonsToRemove: ["toImage"]});
                if (!valid(ticket) || currentJob !== jobId) { Plotly.purge(graph); return; }
                graph.on("plotly_click", event => {
                    if (!valid(ticket) || currentJob !== jobId) return;
                    const point = event.points?.[0]?.customdata; if (!point) return;
                    const image = node("img"), link = node("a", "Open image →");
                    image.src = `/api/hosted/analysis/jobs/${jobId}/images/${encodeURIComponent(point.hosted_identity.id)}/media`; image.alt = point.filename;
                    link.href = `/annotate?image=${encodeURIComponent(point.hosted_identity.id)}`;
                    selected.replaceChildren(image, node("strong", point.filename), node("p", point.kind === "candidate" ? "Incoming image · awaiting annotation and approval" : "Confirmed reference image"), link);
                });
                disposeGraph = () => Plotly.purge(graph);
                content.append(node("small", "The original ingestion analysis fits these PCA axes on confirmed images and projects Incoming images onto the same axes. The selection uses the full descriptors; this plot is a view of them."));
            }
            const grid = node("div"); grid.className = "hosted-caption-proposals"; content.append(grid);
            let offset = 0;
            function page() {
                grid.querySelector("button")?.remove();
                for (const item of items.slice(offset, offset + 24)) {
                    if (!item.hosted_identity) continue;
                    const card = node("article"), image = node("img"), link = node("a", "Inspect and annotate →");
                    image.src = `/api/hosted/analysis/jobs/${jobId}/images/${encodeURIComponent(item.hosted_identity.id)}/media`; image.alt = item.filename; image.loading = "lazy";
                    link.href = `/annotate?image=${encodeURIComponent(item.hosted_identity.id)}`;
                    const scores = [`Incoming diversity: ${Number(item.diversity_score).toFixed(3)}`];
                    if (Number.isFinite(item.reference_novelty_score)) scores.unshift(`Reference novelty: ${item.reference_novelty_score.toFixed(3)}`);
                    card.append(image, node("strong", item.keep ? `Suggested addition · rank ${item.rank}` : "Lower priority in this selection"), node("p", scores.join(" · ")), node("small", item.filename), link); grid.append(card);
                }
                offset += 24; if (offset < items.length) act(grid, "Show more incoming images", page);
            }
            page();
        } else if (job.consumer === "salad_training") {
            const head = result.result?.summary || result.summary || {};
            content.append(node("p", `SALAD training complete${head.label ? ": " + head.label : ""}. The weights and training settings are retained with this snapshot.`));
            act(content, "Register descriptor in Analysis presets", () => { dialog.close(); document.getElementById("hostedProfiles").click(); });
        } else if (job.consumer === "dqe_review") {
            content.append(node("p", "Qwen's visual review is a proposal. Inspect the image and its evidence before making a correction."));
            const pre = node("pre", JSON.stringify(result.result || result, null, 2)); pre.className = "hosted-dqe-evidence"; content.append(pre);
        } else {
            content.append(node("p", "The completed model, logs and recipe are retained with this snapshot."));
            // Training runs record the train/validation split they used.
            const split = result.run_split;
            const placed = split ? split.train_images + split.validation_images + split.test_images - split.images_split_for_run : 0;
            if (split) content.append(node("p", `This run trained on ${count(split.train_images, "image")} and validated on ${count(split.validation_images, "image")}${split.test_images ? `, leaving ${count(split.test_images, "test image")} out` : ""}. `
                + (placed > 0 ? `${count(placed, "image")} kept the split set on the dataset; the other ${count(split.images_split_for_run, "image")} ${split.images_split_for_run === 1 ? "was" : "were"} split for this run only.`
                    : "The dataset has no splits, so this run chose them; the dataset is not changed.")));
            else if (trainingTasks.has(job.consumer)) content.append(node("p", "This run used the splits set on the dataset."));
            // Trainers fall back to the CPU when the configured GPU is not available; say where it ran.
            const device = (result.status?.logs || []).map(entry => String(entry?.message || "")).find(message => message.startsWith("Training device:"));
            if (device) content.append(node("p", `${device}.`));
        }
        const download = node("a", "Download result and provenance"); download.href = `/api/hosted/analysis/jobs/${jobId}/result`; download.download = `${job.consumer}-${jobId}.json`; content.append(download);
    }
    let disposeGraph = null;
    async function showGraph(jobId, ticket) {
        disposeGraph?.(); disposeGraph = null;
        const cleanup = await window.TatorHostedDQE({app, jobId, datasetId, container: content,
            valid: () => valid(ticket) && currentJob === jobId, report: error});
        if (valid(ticket) && currentJob === jobId) disposeGraph = cleanup;
        else cleanup?.();
    }
    let openedLink = false;
    window.addEventListener("tator:hosted-workspace", async () => {
        const query = new URLSearchParams(location.search), jobId = query.get("analysis");
        if (openedLink || !jobId || app.session?.user.role !== "admin") return;
        openedLink = true;
        try {
            const job = await api(`/analysis/jobs/${encodeURIComponent(jobId)}`);
            // Open the result in its own dataset, and keep the address in step with it.
            await app.selectDataset(job.dataset_id);
            app.dropRouteParams?.("analysis", "panel");
            if (query.get("panel") === "profiles") document.getElementById("hostedProfiles")?.click();
            else { await open(job.consumer); await showResult(jobId, epoch); }
        } catch (e) { app.dropRouteParams?.("analysis", "panel"); if (!e.stale) app.notice(e.message, true); }
    });
    window.TatorHostedAnalysis = {open};
    window.addEventListener("tator:caption-export", () => {
        if (app.session?.user.role !== "admin") return;
        const ticket = epoch + 1;
        open("caption_export").catch(e => { if (valid(ticket) && !e.stale) error(e); });
    });
    dialog.addEventListener("close", () => { ++epoch; clearTimeout(timer); disposeGraph?.(); disposeGraph = null; });
    window.addEventListener("tator:hosted-session", () => {
        if (app.session?.user.role !== "admin") { ++epoch; clearTimeout(timer); dialog.close(); controls.replaceChildren(); history.replaceChildren(); content.replaceChildren(); }
    });
})();
