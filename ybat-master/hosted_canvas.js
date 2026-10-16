/* Hosted bridge for the existing first-pane canvas. Authority stays on the server. */
(() => {
    "use strict";
    const boot = window.TatorHostedBoot;
    if (!boot) return;
    const nativeFetch = window.fetch.bind(window);
    const editorId = crypto.randomUUID();
    let accountChanged = false;
    let imageId = boot.image_id;
    let base = `/api/hosted/images/${encodeURIComponent(imageId)}`;
    let key = `tator.hosted.draft:${boot.instance_id}:${boot.user.user_id}:${imageId}`;
    let record, dataset, editable = false, ready = false, pending = null, saving = null, lastSaved = "", busy = false, handoff = false, recoveryRequired = false, toolState = null, retentionCommand = null;
    let rejected = null, storageFailed = false;
    const $ = (id) => document.getElementById(id);
    const uuid = () => crypto.randomUUID();
    const toolRoutes = { "/sam_slots": "sam_state", "/sam_preload": "sam_preload", "/sam_point": "sam_point", "/sam_point_multi": "sam_point_multi", "/sam_bbox": "sam_bbox",
        "/qwen/infer": "qwen_infer", "/qwen/caption": "qwen_caption",
        "/qwen/prepass": "edr_prepass",
        "/agent_mining/apply_image_chain": "sam3_recipe_chain",
        "/qwen/caption/preview_prompt": "qwen_caption_preview",
        "/sam_point_auto": "sam_point_auto", "/sam_point_multi_auto": "sam_point_multi_auto", "/sam_bbox_auto": "sam_bbox_auto", "/predict_base64": "classify_bbox",
        "/sam3/text_prompt": "sam3_text", "/sam3/visual_prompt": "sam3_visual",
        "/yolo/predict_full": "detect_full", "/rfdetr/predict_full": "detect_full", "/yolo/predict_region": "detect_region", "/rfdetr/predict_region": "detect_region",
        "/yolo/predict_windowed": "detect_windowed", "/rfdetr/predict_windowed": "detect_windowed" };
    const activeTools = new Map();
    const jsonResponse = (value) => new Response(JSON.stringify(value), { status: 200, headers: { "content-type": "application/json" } });
    const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
    const messages = {
        annotation_version_changed: "This image changed on the server. Your unsaved draft is retained. Reload to compare before saving.",
        ownership_changed: "Editing ownership changed. Your draft is retained, but it cannot overwrite the new owner's work.",
        editor_lease_conflict: "This image is open in another editing tab. Close it there or explicitly take over below.",
        editor_lease_expired: "Your editing connection expired. Your draft is retained. Reconnect before continuing.",
        login_required: "Your session ended. Sign in again; your unsent draft is retained for your account.",
        csrf_rejected: "Your account session changed in another tab. Reconnect before continuing.",
        installation_changed: "The server was restored or replaced. Reconnect before editing.",
        image_owned_by_annotator: "The assigned annotator is editing this image. You can view it, or explicitly recover it from their assignment.",
        negative_has_objects: "This image contains annotations. Clear the empty-image check before saving.",
        instruction_review_requires_admin: "Only an administrator can change review decisions. Your draft is retained.",
        machine_validation_requires_server_evidence: "These QA validation labels do not match a saved server result. Your draft is retained; ask an administrator to review the affected answers.",
        machine_validation_source_changed: "The caption result belongs to an earlier image version or editing session. Your draft is retained; generate it again from the current saved image.",
    };
    function status(text, error = false) { if ($("hostedSaveState")) { $("hostedSaveState").textContent = text; $("hostedSaveState").dataset.error = String(error); } }
    function freeze() {
        // Editing authority and local recovery are separate. Capture the newest
        // draft while the original image/account context is still available.
        let retentionFailure = null;
        try { retainDraft(); } catch (error) { retentionFailure = error; }
        editable = false; window.TatorHostedCanvas?.freeze(); update();
        return retentionFailure;
    }
    function report(error) {
        if (pending && error.commit_outcome === "committed" && error.result_restricted === true && error.operation_id === pending.body.operation_id) {
            retainRestrictedCommit(pending, error);
        }
        const retentionFailure = [401, 403, 409, 423].includes(error.status) ? freeze() : null;
        status(retentionFailure?.message || error.message || String(error), true);
    }
    function headers(options = {}) {
        const h = new Headers(options.headers);
        h.set("x-tator-csrf", boot.csrf); h.set("x-tator-instance", boot.instance_id);
        return h;
    }
    async function api(path, body, method = body === undefined ? "GET" : "POST") {
        if (accountChanged) throw Object.assign(new Error("Your account changed in another tab. Reopen the workspace to continue."), {status: 401});
        const options = { method, credentials: "same-origin", headers: headers() };
        if (body !== undefined) { options.headers.set("content-type", "application/json"); options.body = JSON.stringify(body); }
        const response = await nativeFetch(path, options);
        let value;
        try { value = await response.json(); }
        catch (_) { throw new Error("The server reply was incomplete. Your pending change is retained; reconnect or save again to check its original result."); }
        if (accountChanged) throw Object.assign(new Error("Your account changed while this request was running. Its result is retained for the original account."), {status: 401});
        if (!response.ok) { const error = new Error(messages[value.error] || (typeof value.detail === "string" ? value.detail : value.error) || `Request failed (${response.status}).`); error.status = response.status; error.code = value.error;
            for (const name of ["commit_outcome", "operation_id", "result_restricted"]) error[name] = value[name];
            throw error; }
        return value;
    }
    function checkedResult(operation, result) {
        const kind = operation.path.split("/").at(-1);
        const state = {submit: "submitted", approve: "confirmed", return: "returned"}[kind];
        const valid = kind === "archive"
            ? result?.kind === "image" && result.target_id === imageId && result.state === "archived" && typeof result.archive_id === "string"
            : result?.id === imageId && typeof result.current_version === "string" && Number.isInteger(result.generation) && (!state || result.state === state);
        if (!valid) throw new Error("The server reply was incomplete. Your pending change is retained; check its original result before continuing.");
        return result;
    }
    async function commandReceipt(operation) {
        const receipt = await api(`/api/hosted/receipts/${encodeURIComponent(operation.body.operation_id)}`);
        if (receipt === null) return null;
        if (receipt?.operation_id !== operation.body.operation_id) throw new Error("The saved-operation receipt was incomplete. Your pending change is retained.");
        if (receipt.commit_outcome === "committed" && receipt.result_restricted === true) {
            if (operation.path.endsWith("/save")) {
                retainRestrictedCommit(operation, receipt);
                throw new Error("Your earlier change was saved, but your account can no longer view its result. Any later draft is retained in this tab. Return to the workspace or ask an administrator for access.");
            }
            return receipt;
        }
        checkedResult(operation, receipt.result);
        return receipt;
    }
    function retainRestrictedCommit(operation, receipt) {
        let previous = null;
        try { previous = JSON.parse(sessionStorage.getItem(key) || "null"); } catch (_) { /* Keep the live draft if storage failed. */ }
        const latest = ready && record && !recoveryRequired ? draft() : previous?.latest;
        pending = null; retentionCommand = null; recoveryRequired = true; editable = false; handoff = true;
        window.TatorHostedCanvas?.freeze();
        journal({ state: "committed_restricted", committed: { operation_id: operation.body.operation_id, commit_outcome: "committed", result_restricted: true },
            base: { expected_version: operation.body.expected_version, generation: operation.body.generation }, latest });
        for (const id of ["hostedHandoffRetry", "hostedImportConflictRetry", "hostedInstructionReviewRetry", "hostedPaidCaptionRetry", "hostedCaptionRetry"]) $(id)?.remove();
        update();
    }
    async function toolFetch(operation, options) {
        if (!record || !dataset) throw new Error("Load an assigned image before using its tools.");
        const source = typeof options.body === "string" ? JSON.parse(options.body) : {};
        const requestedImage = source.image_name ? source.image_name.split("/").at(-1) : imageId;
        const origin = {imageId, datasetId: dataset.id, generation: record.generation};
        const readOnly = ["sam_preload", "sam_state"].includes(operation);
        if (requestedImage !== imageId && !readOnly) throw new Error("Open this image before requesting its tools.");
        if (!readOnly && !window.TatorHosted.editable) throw new Error("Reconnect editing before using this tool.");
        if (!readOnly && !await save({ manual: true })) throw new Error("Save this image before requesting its tools.");
        let accepted = ["point_x", "point_y", "positive_points", "negative_points", "bbox_left", "bbox_top", "bbox_width", "bbox_height", "local_refine", "uuid", "slot", "conf", "iou", "max_det", "slice_size", "overlap", "merge_iou", "text_prompt", "prompt", "threshold", "mask_threshold", "min_size", "max_results", "windowed", "window_size", "window_overlap", "simplify_epsilon", "bboxes", "bbox_labels"];
        if (operation === "edr_prepass") accepted = ["recipe_id"];
        if (operation.startsWith("qwen_") || operation === "sam3_recipe_chain") {
            const catalog = await api(`/api/hosted/tools/catalog?dataset_id=${encodeURIComponent(dataset.id)}`);
            accepted = catalog.parameters?.[operation];
            if (!Array.isArray(accepted)) throw new Error("This server does not expose the requested tool.");
        }
        const parameters = Object.fromEntries(accepted.filter((key) => source[key] !== undefined).map((key) => [key, source[key]]));
        if (source.region) [parameters.bbox_left, parameters.bbox_top, parameters.bbox_width, parameters.bbox_height] = source.region;
        if (operation === "classify_bbox") {
            if (!Array.isArray(source.bbox_xyxy) || source.bbox_xyxy.length !== 4) throw new Error("The classifier needs the object's position in this image. Draw the box again.");
            const [left, top, right, bottom] = source.bbox_xyxy;
            Object.assign(parameters, { bbox_left: left, bbox_top: top, bbox_width: right - left, bbox_height: bottom - top });
        }
        if (source.slice_overlap !== undefined) parameters.overlap = source.slice_overlap;
        const targetRecord = requestedImage === imageId ? record : await api(`/api/hosted/images/${encodeURIComponent(requestedImage)}`);
        if (!readOnly && (imageId !== origin.imageId || dataset.id !== origin.datasetId || record.generation !== origin.generation || !window.TatorHosted.editable)) throw new Error("The image changed while preparing this tool. Run it again on the current image.");
        if (targetRecord.dataset_id !== dataset.id) throw new Error("The requested image belongs to another dataset.");
        const captured = { dataset_id: dataset.id, image_id: targetRecord.id, expected_version: targetRecord.current_version,
            generation: targetRecord.generation, editor_id: editorId, workspace_id: editorId, operation, parameters };
        options.signal?.throwIfAborted();
        const request = await api("/api/hosted/tools/request", captured);
        activeTools.set(request.id, { ...request, operation });
        const cancel = () => api(`/api/hosted/tools/requests/${request.id}/cancel`, {}).catch(() => {});
        options.signal?.addEventListener("abort", cancel, { once: true });
        try {
            if (options.signal?.aborted) await cancel();
            let state = request;
            while (!["completed", "failed", "cancelled", "stale", "interrupted"].includes(state.state)) {
                options.signal?.throwIfAborted();
                await new Promise((resolve) => setTimeout(resolve, 300));
                state = await api(`/api/hosted/tools/requests/${request.id}`);
                activeTools.set(request.id, { ...state, operation });
            }
            if (state.state !== "completed") throw new Error(`Tool ${state.state}: ${state.error || "please retry on the current image"}`);
            const result = await api(`/api/hosted/tools/requests/${request.id}/result`);
            options.signal?.throwIfAborted();
            if ((captured.image_id === imageId && (record.current_version !== captured.expected_version || record.generation !== captured.generation)) || (!readOnly && (captured.image_id !== imageId || !window.TatorHosted.editable))) {
                throw new Error("The image changed while the tool was running. Run it again on the current version.");
            }
            const output = result.result;
            if (operation === "sam_state") return jsonResponse(output.slots.map(slot => ({...slot, image_name: slot.image_id ? window.TatorHostedCanvas.imageNameFor(slot.image_id) : null})));
            for (const item of output.detections || [output]) {
                if (!item.stable_class_id) continue;
                const currentClass = dataset.classes.find((entry) => entry.id === item.stable_class_id && !entry.retired);
                if (!currentClass) throw new Error("The dataset's classes changed. Reopen this image before applying predictions.");
                item.class_id = currentClass.position;
                if (item.label !== undefined) item.label = currentClass.name;
                if (item.prediction !== undefined) item.prediction = currentClass.name;
            }
            return new Response(JSON.stringify(output), { status: 200, headers: { "content-type": "application/json" } });
        } finally { options.signal?.removeEventListener("abort", cancel); activeTools.delete(request.id); }
    }
    const captionJobContexts = new Map();
    async function captionJobFetch(target, options) {
        const path = target.pathname, prefix = "/qwen/caption/jobs";
        const suffix = path.slice(prefix.length), method = (options.method || "GET").toUpperCase();
        const source = typeof options.body === "string" ? JSON.parse(options.body) : {};
        if ((!suffix || suffix === "/preview_process") && method === "POST") {
            if (!window.TatorHosted.editable) throw new Error("Reconnect editing before generating captions.");
            const origin = {imageId, datasetId: dataset.id, generation: record.generation};
            const names = source.image_names || [];
            if (!names.length || (suffix !== "/preview_process" && names.length !== 1) || names[0].split("/").at(-1) !== imageId) throw new Error("This caption action needs one open assigned image. Dataset-wide caption jobs belong in the administrator workspace.");
            if (source.caption_provider && !["qwen", "local_qwen", "openai"].includes(source.caption_provider)) throw new Error("Choose a caption provider configured for this dataset.");
            if (source.require_pilot_certification) throw new Error("A filesystem pilot cannot authorize this hosted image job. Turn off dataset pilot certification for a single assigned image.");
            if (!await save({ manual: true })) throw new Error("Save this image before generating captions.");
            const catalog = await api(`/api/hosted/tools/catalog?dataset_id=${encodeURIComponent(dataset.id)}`);
            if (imageId !== origin.imageId || dataset.id !== origin.datasetId || record.generation !== origin.generation || !window.TatorHosted.editable) throw new Error("The image changed while preparing captions. Start again on the current image.");
            const fields = catalog.caption_job_parameters;
            if (!fields) throw new Error("This server does not expose hosted caption jobs.");
            const pick = (value, keys) => Object.fromEntries(keys.filter((key) => value[key] !== undefined).map((key) => [key, value[key]]));
            const captured = { dataset_id: dataset.id, image_id: imageId, ...context(), workspace_id: editorId,
                caption_request: pick(source.caption_request || {}, fields.caption_request), settings: pick(source, fields.settings) };
            if (source.caption_provider === "openai") {
                const model = catalog.paid_caption?.models.find(item => item.id === source.openai_model);
                if (!model) throw new Error("The administrator has not enabled this paid model for this dataset.");
                if (source.openai_service_tier === "batch") throw new Error("Use the remote Batch controls for an asynchronous batch, or choose Standard for this caption flow.");
                captured.provider = "openai";
                captured.provider_options = {model: model.id, detail: source.openai_image_detail || "high", reasoning_effort: source.openai_reasoning_effort || null,
                    max_output_tokens: captured.caption_request.max_new_tokens || Math.min(2048, model.max_output_tokens), service_tier: "standard"};
                // Local sampling controls are visibly disabled for this provider.
                for (const key of ["temperature", "top_p", "top_k", "presence_penalty"]) delete captured.caption_request[key];
                for (const key of ["model_id", "refinement_model_id", "caption_fallback_model_id"]) captured.caption_request[key] = model.id;
            }
            if (suffix === "/preview_process") captured.settings.preview_only = true;
            let job;
            try { job = await api("/api/hosted/tools/caption-jobs", captured); }
            catch (error) { if (error.status) throw error; job = await api("/api/hosted/tools/caption-jobs", captured); }
            captionJobContexts.set(job.job_id, captured);
            activeTools.set(job.job_id, { id: job.job_id, operation: "qwen_caption_job", state: job.status });
            if (options.signal?.aborted) await api(`/api/hosted/tools/caption-jobs/${job.job_id}/cancel`, { operation_id: uuid() });
            if (suffix === "/preview_process") {
                const cancel = () => api(`/api/hosted/tools/caption-jobs/${job.job_id}/cancel`, { operation_id: uuid() }).catch(() => {});
                options.signal?.addEventListener("abort", cancel, { once: true });
                try {
                    while (!["completed", "failed", "cancelled", "interrupted", "waiting_for_editor"].includes(job.status)) {
                        options.signal?.throwIfAborted();
                        await new Promise((resolve) => setTimeout(resolve, 300));
                        job = await api(`/api/hosted/tools/caption-jobs/${job.job_id}`);
                    }
                    options.signal?.throwIfAborted();
                    if (job.status === "waiting_for_editor") throw new Error("Caption preview paused. Reconnect editing on this image to resume.");
                    if (job.status !== "completed") throw new Error(job.error || "Caption preview did not finish.");
                    return jsonResponse(job.result.preview);
                } finally { options.signal?.removeEventListener("abort", cancel); activeTools.delete(job.job_id); }
            }
            for (const row of [...(job.result?.captions || []), job.result?.latest_caption].filter(Boolean)) row.image_name = window.TatorHostedCanvas.imageName();
            if (["completed", "failed", "cancelled", "interrupted", "waiting_for_editor"].includes(job.status)) activeTools.delete(job.job_id);
            return jsonResponse(job);
        }
        if (!suffix && method === "GET") return jsonResponse(await api("/api/hosted/tools/caption-jobs" + target.search));
        const match = suffix.match(/^\/([a-zA-Z0-9_-]+)(?:\/(cancel|result|resume))?$/);
        if (!match) throw new Error("This caption job action is not available in the hosted workspace.");
        const [, id, action] = match;
        if (action === "resume") {
            if (!window.TatorHosted.editable) throw new Error("Reconnect editing before resuming captions.");
            const origin = {imageId, generation: record.generation};
            if (!await save({ manual: true })) throw new Error("Save this image before resuming captions.");
            const previous = await api(`/api/hosted/tools/caption-jobs/${id}`);
            if (imageId !== origin.imageId || record.generation !== origin.generation || !window.TatorHosted.editable) throw new Error("The image changed while preparing captions. Reopen the original image to resume.");
            if (!previous.request || previous.context?.image_id !== imageId || previous.context?.expected_version !== record.current_version) throw new Error("This caption job describes a different image version. Start a new job with the current annotations.");
            const body = { ...previous.request, ...context(), workspace_id: editorId };
            const resumed = await api(`/api/hosted/tools/caption-jobs/${id}/resume`, body);
            captionJobContexts.set(id, body); activeTools.set(id, { id, operation: "qwen_caption_job", state: resumed.status });
            return jsonResponse(resumed);
        }
        const job = await api(`/api/hosted/tools/caption-jobs/${id}${action ? "/" + action : ""}`,
            action === "cancel" ? { operation_id: uuid() } : undefined);
        const state = job.status || job.state;
        if (["completed", "failed", "cancelled", "interrupted", "waiting_for_editor"].includes(state)) activeTools.delete(id);
        else if (captionJobContexts.has(id)) activeTools.set(id, { id, operation: "qwen_caption_job", state });
        const captured = job.context || captionJobContexts.get(id);
        if (state === "completed" && captured && (captured.image_id !== imageId || captured.expected_version !== record.current_version || captured.generation !== record.generation || !window.TatorHosted.editable)) {
            throw new Error("This caption belongs to an earlier editing version. It has been retained but cannot replace the current image's caption.");
        }
        if (captured?.image_id === imageId) {
            for (const row of [...(job.result?.captions || []), job.result?.latest_caption].filter(Boolean)) row.image_name = window.TatorHostedCanvas.imageName();
        }
        return jsonResponse(job);
    }
    async function qwenMetadata(path) {
        const current = [...activeTools.values()].find((item) => item.operation.startsWith("qwen_") || item.operation === "edr_prepass");
        if (["/qwen/cancel", "/qwen/caption/cancel"].includes(path)) {
            const targets = [...activeTools.values()].filter((item) => path === "/qwen/cancel" ? (item.operation.startsWith("qwen_") || item.operation === "edr_prepass") : ["qwen_caption", "qwen_caption_job"].includes(item.operation));
            await Promise.all(targets.map((item) => item.operation === "qwen_caption_job"
                ? api(`/api/hosted/tools/caption-jobs/${item.id}/cancel`, { operation_id: uuid() })
                : api(`/api/hosted/tools/requests/${item.id}/cancel`, {})));
            return jsonResponse({ cancelled: targets.length });
        }
        const progress = { active: Boolean(current), run_id: current?.id || null, phase: current?.state || "idle",
            phase_label: current ? (current.state === "running" ? "Processing on your private worker" : "Waiting for your private worker") : "Idle",
            message: current ? "The result will appear when processing finishes." : "", token_preview: "" };
        if (path === "/qwen/progress") return jsonResponse(progress);
        if (!dataset) return jsonResponse({ available: false, loaded: false, models: [], progress });
        const catalog = await api(`/api/hosted/tools/catalog?dataset_id=${encodeURIComponent(dataset.id)}`);
        const state = await api(`/api/hosted/tools/state?dataset_id=${encodeURIComponent(dataset.id)}`);
        const models = catalog.models.filter((item) => item.kind === "qwen" && catalog.qwen?.allowed_model_ids?.includes(item.model_id));
        if (path === "/qwen/models") return jsonResponse({ active: catalog.qwen?.default_model_id, models: models.map((item) => ({ id: item.model_id, label: item.model_id, active: item.model_id === catalog.qwen.default_model_id, variant: item.variant,
            available: true, inference_supported: true, local: true, metadata: { label: item.model_id, model_family: "qwen3", variant: item.variant, runtime_platform: "transformers" } })), progress });
        return jsonResponse({ available: Boolean(state.capabilities?.qwen_infer?.configured) && state.phase === "available", loaded: false, effective_model_name: catalog.qwen?.default_model_id, progress });
    }
    const metadataRoutes = {
        "/clip/classifiers": "classifiers", "/clip/labelmaps": "labelmaps", "/clip/active_model": "active_classifier",
        "/sam3/models/available": "sam_models", "/agent_mining/recipes": "sam3_recipes", "/prepass/recipes": "prepass_recipes",
    };
    const cascadeRevisions = new Map();
    async function installedToolsFetch(target, options) {
        if (!dataset) throw new Error("Open a dataset before choosing its tools.");
        const capturedDataset = dataset.id, path = target.pathname, method = (options.method || "GET").toUpperCase();
        const metadata = () => api(`/api/hosted/tools/metadata?dataset_id=${encodeURIComponent(capturedDataset)}`);
        if (metadataRoutes[path] && method === "GET") {
            let value = (await metadata())[metadataRoutes[path]];
            if (path === "/sam3/models/available") {
                const variant = target.searchParams.get("variant");
                if (variant && variant !== "all") value = value.filter((item) => item.variant === variant);
            }
            return jsonResponse(value);
        }
        if (path === "/sam3/models/activate" && method === "POST") {
            const source = JSON.parse(options.body || "{}"), current = await metadata();
            if (!current.sam_models.some((item) => item.id === source.checkpoint_path)) throw new Error("Choose an installed SAM model permitted for your account.");
            const selected = await api("/api/hosted/tools/sam", { model_id: source.checkpoint_path,
                expected_revision: current.sam_selection.revision, operation_id: uuid() });
            return jsonResponse({ active: selected.model_id, model_id: selected.model_id, configured: true });
        }
        if (path.startsWith("/agent_mining/cascades")) {
            const base = `/api/hosted/tools/datasets/${encodeURIComponent(capturedDataset)}/cascades`;
            const suffix = path.slice("/agent_mining/cascades".length);
            const remember = (value) => {
                for (const row of Array.isArray(value) ? value : [value]) cascadeRevisions.set(`${capturedDataset}:${row.id}`, row.revision);
                return jsonResponse(value);
            };
            const savePreset = (value) => api(base, { label: value.label, steps: value.steps, dedupe: value.dedupe,
                expected_revision: 0, operation_id: uuid() });
            if (!suffix && method === "GET") return remember(await api(base));
            if (!suffix && method === "POST") return remember(await savePreset(JSON.parse(options.body || "{}")));
            if (suffix === "/import" && method === "POST") {
                const file = options.body instanceof FormData ? options.body.get("file") : null;
                if (!(file instanceof Blob) || file.size > 1024 * 1024) throw new Error("Choose a hosted cascade preset ZIP smaller than 1 MB. Model packages must be installed by an administrator.");
                if (!window.JSZip) throw new Error("The ZIP reader is unavailable. Reload this page.");
                const zip = await window.JSZip.loadAsync(await file.arrayBuffer());
                const entry = zip.file("hosted-cascade.json");
                // Bound decompression before allocating the exported JSON string.
                if (!entry || !Number.isFinite(entry._data?.uncompressedSize) || entry._data.uncompressedSize > 1024 * 1024) throw new Error("This is not a hosted cascade preset. Install its saved recipes through Tools & models first.");
                const bundle = JSON.parse(await entry.async("string"));
                if (bundle.schema !== "tator.hosted.cascade.v1") throw new Error("This cascade format is not supported.");
                const catalog = await api(`/api/hosted/tools/catalog?dataset_id=${encodeURIComponent(capturedDataset)}`);
                for (const step of bundle.cascade?.steps || []) {
                    if (!bundle.recipe_identities?.[step.recipe_id] || bundle.recipe_identities[step.recipe_id] !== catalog.sam3_recipes?.identities?.[step.recipe_id]) throw new Error("This preset needs the same installed recipe versions. Ask an administrator to attach them before importing.");
                }
                return remember(await savePreset(bundle.cascade));
            }
            const match = suffix.match(/^\/([a-zA-Z0-9_-]+)(?:\/(export))?$/);
            if (!match) throw new Error("This cascade action is not available.");
            const [, id, action] = match, url = `${base}/${encodeURIComponent(id)}`;
            if (method === "GET") {
                const value = await api(url);
                if (action === "export") {
                    if (!window.JSZip) throw new Error("The ZIP writer is unavailable. Reload this page.");
                    const catalog = await api(`/api/hosted/tools/catalog?dataset_id=${encodeURIComponent(capturedDataset)}`);
                    const zip = new window.JSZip();
                    zip.file("hosted-cascade.json", JSON.stringify({ schema: "tator.hosted.cascade.v1", cascade: value,
                        recipe_identities: catalog.sam3_recipes?.identities || {} }, null, 2));
                    return new Response(await zip.generateAsync({ type: "blob", compression: "STORE" }), { headers: { "content-type": "application/zip" } });
                }
                return remember(value);
            }
            if (method === "DELETE" && !action) {
                const revision = cascadeRevisions.get(`${capturedDataset}:${id}`);
                if (revision === undefined) throw new Error("Refresh your cascade list before deleting this preset.");
                return jsonResponse(await api(`${url}/delete`, { expected_revision: revision, operation_id: uuid() }));
            }
        }
        throw new Error("An administrator manages installed model packages in Tools & models. Your cascade presets can be saved and downloaded from the cascade controls.");
    }
    function context() { return { expected_version: record.current_version, generation: record.generation, editor_id: editorId, operation_id: uuid() }; }
    function draft() { return { annotations: window.TatorHostedCanvas.snapshot(), complete: record.state === "confirmed" || Boolean($("hostedComplete")?.checked), negative: Boolean($("hostedNegative")?.checked) }; }
    function journal(value) {
        try { sessionStorage.setItem(key, JSON.stringify({ schema: 2, state: value.pending ? "pending" : value.rejected ? "rejected" : "unsent", ...value })); storageFailed = false; }
        catch (_) { storageFailed = true; throw new Error("This browser cannot retain unsent edits. Keep this tab open and free browser storage before continuing."); }
    }
    function clearJournal() {
        try { sessionStorage.removeItem(key); }
        catch (_) {
            storageFailed = true;
            throw new Error("This browser could not clear its earlier draft. Keep this tab open and restore browser storage before leaving.");
        }
        retentionCommand = null; rejected = null; storageFailed = false;
    }
    function reconcileCleanDraft(latest) {
        // Only the live, acknowledged canvas can supersede an obsolete draft.
        // A server canvas loaded during recovery says nothing about that draft.
        if (!ready || !record || recoveryRequired || pending || saving || JSON.stringify(latest) !== lastSaved) return false;
        clearJournal();
        return true;
    }
    function retainDraft() {
        // During recovery the canvas initially shows the server version, not the
        // retained draft. Do not overwrite that journal before explicit restore.
        if (!ready || !record || recoveryRequired) return;
        const latest = draft();
        if (!pending && JSON.stringify(latest) === lastSaved) { reconcileCleanDraft(latest); return; }
        if (!pending && (!retentionCommand || retentionCommand.path !== `${base}/save`
            || retentionCommand.body.expected_version !== record.current_version || retentionCommand.body.generation !== record.generation)) {
            retentionCommand = { path: `${base}/save`, body: { ...context(), ...latest } };
        }
        try {
            journal({ pending, unsent: pending ? null : retentionCommand, rejected, latest });
        } catch (_) {
            throw new Error("This browser could not retain your latest unsent edits. Keep this tab open and free browser storage before leaving.");
        }
    }
    function update() {
        for (const id of ["hostedSave", "hostedSubmit", "hostedApprove", "hostedReturn", "hostedArchive", "hostedComplete", "hostedNegative"]) {
            if ($(id)) $(id).disabled = !editable || !ready || busy || handoff;
        }
        if ($("hostedReconnect")) $("hostedReconnect").hidden = editable;
        if ($("hostedTakeover")) $("hostedTakeover").hidden = editable || recoveryRequired || (boot.user.role === "admin" && ["assigned", "returned"].includes(record.state));
    }
    async function save({ manual = false } = {}) {
        if (saving) { await saving; if (manual) return save({ manual }); return false; }
        if (!ready || !editable || handoff) return false;
        if (pending && !pending.path.endsWith("/save")) throw new Error("Resolve the earlier handoff before saving again.");
        const current = draft();
        if (!pending && JSON.stringify(current) === lastSaved) {
            try { reconcileCleanDraft(current); } catch (error) { report(error); throw error; }
            if (manual) status("All changes saved"); return true;
        }
        if (!pending && rejected && JSON.stringify(current) === rejected.draft) {
            if (manual) throw new Error(rejected.message);
            return false;
        }
        if (!pending) rejected = null;
        if (!pending) pending = { path: `${base}/save`, body: { ...context(), ...current } };
        const sent = pending;
        journal({ pending: sent, latest: current });
        status("Saving…");
        saving = (async () => {
            const result = checkedResult(sent, await api(sent.path, sent.body));
            record = { ...record, ...result, annotations: structuredClone(sent.body.annotations) };
            window.TatorHostedCanvas.acknowledge(record);
            lastSaved = JSON.stringify({ annotations: sent.body.annotations, complete: sent.body.complete, negative: sent.body.negative });
            pending = null;
            const latest = draft();
            if (JSON.stringify(latest) === lastSaved) { clearJournal(); status("All changes saved"); }
            else { retentionCommand = { path: `${base}/save`, body: { ...context(), ...latest } }; journal({ unsent: retentionCommand, latest }); status("Earlier edits saved · saving remaining changes next"); }
            return true;
        })();
        try { return await saving; } catch (error) {
            if (error.status === 422 && error.commit_outcome === "rejected" && error.operation_id === sent.body.operation_id) {
                // The server certifies this exact transaction rolled back. Edits
                // made while awaiting its reply belong to the next command.
                rejected = { operation_id: sent.body.operation_id, context: { expected_version: sent.body.expected_version, generation: sent.body.generation },
                    draft: JSON.stringify({ annotations: sent.body.annotations, complete: sent.body.complete, negative: sent.body.negative }), message: error.message };
                pending = null; retentionCommand = null;
                try { journal({ rejected, latest: draft() }); } catch (retentionError) { report(retentionError); throw retentionError; }
            }
            report(error); throw error;
        }
        finally { saving = null; update(); }
    }
    async function run(button, action) { if (busy) return; busy = true; update(); try { await action(); } catch (error) { report(error); } finally { busy = false; update(); } }
    window.addEventListener("storage", event => {
        if (event.key === "tator.hosted.session-event") {
            accountChanged = true;
            const retentionFailure = freeze();
            status(retentionFailure?.message || "Your account changed in another tab. Unsent edits are retained for the original account. Reopen the workspace to continue.", true);
        }
    });
    async function verifySession() {
        const current = await api("/api/hosted/session");
        if (current.user.user_id !== boot.user.user_id || current.instance_id !== boot.instance_id || current.csrf !== boot.csrf) {
            const error = new Error("Your login changed. Reopen the workspace to continue with the correct account."); error.status = 401; throw error;
        }
    }
    async function acquire(takeover = false, recovering = false, preparedRecord = null, sessionCheck = null) {
        if (recoveryRequired && !recovering) throw new Error("Recover the retained draft before reconnecting editing.");
        // Startup/navigation can overlap their session check with independent
        // reads. Only that in-flight transition reuses it; reconnects check anew.
        // The editor request still verifies auth and version/generation itself.
        const [, fresh] = await Promise.all([sessionCheck || verifySession(), preparedRecord ? Promise.resolve(preparedRecord) : api(`${base}?include_review=true`)]);
        if (record && (record.current_version !== fresh.current_version || record.generation !== fresh.generation)) {
            const error = new Error("The server image changed. Reopen it to review the latest version before editing."); error.status = 409; throw error;
        }
        const lease = await api(`${base}/editor`, { editor_id: editorId, expected_version: fresh.current_version, generation: fresh.generation, takeover });
        record = { ...fresh, ...lease }; editable = true;
        window.TatorHostedCanvas?.freeze(false); status("Connected · edits save automatically"); update();
    }
    async function leave(next = false, destination = "/") {
        if (editable && !handoff) { if (!await save({ manual: true })) throw new Error("Save your work before leaving."); await api(`${base}/release`, context()); editable = false; }
        if (next && dataset) {
            const result = await api(`/api/hosted/images?dataset_id=${encodeURIComponent(dataset.id)}&limit=100${boot.user.role === "admin" ? "&state=submitted" : ""}`);
            const image = result.images.find((item) => item.id !== imageId && (boot.user.role === "admin" || ["assigned", "returned"].includes(item.state)));
            if (image) { location.assign(`/annotate?image=${encodeURIComponent(image.id)}`); return; }
        }
        location.assign(destination);
    }
    async function navigate(targetId) {
        if (busy || handoff) throw new Error("Wait for the current save or handoff before changing images.");
        if (targetId === imageId) return;
        if ([...activeTools.values()].some((item) => !["sam_preload", "sam_state"].includes(item.operation))) throw new Error("Wait for the current annotation tool, or cancel it before changing images.");
        busy = true; update();
        try {
            const sessionCheck = verifySession();
            const [, target] = await Promise.all([sessionCheck, api(`/api/hosted/images/${encodeURIComponent(targetId)}?include_review=true`)]);
            if (target.dataset_id !== dataset.id) throw new Error("Open the other dataset from the workspace.");
            if (editable) {
                if (!await save({ manual: true })) throw new Error("Save your current image before changing images.");
                await api(`${base}/release`, context()); editable = false;
            }
            const targetKey = `tator.hosted.draft:${boot.instance_id}:${boot.user.user_id}:${targetId}`;
            if (sessionStorage.getItem(targetKey) || target.state !== record.state) { location.assign(`/annotate?image=${encodeURIComponent(targetId)}`); return; }
            ready = false; editable = false; pending = null; rejected = null; retentionCommand = null;
            imageId = targetId; base = `/api/hosted/images/${encodeURIComponent(imageId)}`; key = targetKey; record = target;
            let accessError = null;
            try { await acquire(false, false, target, sessionCheck); } catch (error) { accessError = error; editable = false; }
            history.replaceState(null, "", `/annotate?image=${encodeURIComponent(imageId)}`);
            $("hostedComplete").checked = record.complete; $("hostedNegative").checked = record.negative;
            document.querySelector(".hosted-canvas-title strong").textContent = record.filename;
            document.querySelectorAll(".hosted-review-note").forEach((item) => item.remove());
            window.TatorHostedCanvas.activate(record, dataset);
            const began = Date.now();
            while (!window.TatorHostedCanvas.ready()) {
                if (Date.now() - began > 30000) throw new Error("This image did not finish loading. Reopen it to retry.");
                await new Promise((resolve) => setTimeout(resolve, 16));
            }
            ready = true; lastSaved = JSON.stringify(draft());
            window.TatorHostedCanvas.freeze(!editable);
            window.dispatchEvent(new Event("tator:hosted-image"));
            history.replaceState(null, "", `/annotate?image=${encodeURIComponent(imageId)}`);
            await refreshFeedback();
            if (accessError) report(accessError);
            else status("Connected · edits save automatically");
        } catch (error) { report(error); throw error; }
        finally { busy = false; update(); }
    }
    async function transition(kind) {
        if (window.TatorHostedCanvas.hasPendingTools() || [...activeTools.values()].some((item) => !["sam_preload", "sam_state"].includes(item.operation))) throw new Error("Wait for the annotation tool to finish, or cancel it before handing over this image.");
        if (pending && !pending.path.endsWith("/save")) throw new Error("Check the earlier handoff before submitting again.");
        if (!["return", "archive"].includes(kind)) {
            if (!$("hostedComplete").checked && !$("hostedNegative").checked) throw new Error("Confirm that you checked the annotations, or mark this as an empty image.");
        }
        if (!await save({ manual: true })) throw new Error("Save before handing over this image.");
        const body = context();
        if (kind === "return") { body.feedback = $("hostedFeedback").value.trim(); if (!body.feedback) throw new Error("Explain what needs changing before returning the image."); }
        if (kind === "archive") body.reason = "Archived by an administrator from the annotation canvas.";
        handoff = true; update();
        const operation = { path: `${base}/${kind}`, body };
        pending = operation;
        journal({ pending: operation, latest: draft() });
        try { checkedResult(operation, await api(operation.path, operation.body)); pending = null; clearJournal(); editable = false; await leave(kind !== "archive"); }
        catch (error) {
            // A transport failure says nothing about whether the server committed.
            // Retain the exact command until its receipt or replay resolves it.
            if (!$("hostedHandoffRetry")) addButton($("hostedSaveState").parentElement, "hostedHandoffRetry", "Check handoff result", async () => {
                const receipt = await commandReceipt(operation);
                if (!receipt) checkedResult(operation, await api(operation.path, operation.body));
                pending = null; clearJournal(); editable = false; await leave(kind !== "archive");
            });
            throw error;
        }
    }
    function addButton(parent, id, label, callback) { const button = node("button", label); button.type = "button"; button.id = id; button.onclick = () => run(button, callback); parent.append(button); return button; }
    function showFeedback(feedback) {
        const notes = $("hostedReviewNotes"); notes.replaceChildren();
        if (feedback) { const note = node("p", `Administrator feedback: ${feedback.feedback}`); note.className = "hosted-review-note"; notes.append(note); }
    }
    async function refreshFeedback() {
        if (Object.prototype.hasOwnProperty.call(record, "review_feedback")) {
            showFeedback(record.review_feedback); return;
        }
        // Public assets can update before the backend restarts. Older servers
        // need the history fallback, but never overwrite a later image/lease.
        const origin = {imageId, generation: record.generation};
        const isCurrent = () => imageId === origin.imageId && record.generation === origin.generation;
        try {
            const history = await api(`${base}/history`);
            if (isCurrent()) showFeedback((history.reviews || []).filter(item => item.feedback).at(-1));
        } catch (error) { if (isCurrent()) throw error; }
    }
    function buildUi() {
        document.body.classList.add("hosted-canvas");
        const panel = node("aside"); panel.className = "hosted-canvas-panel"; panel.setAttribute("aria-label", "Annotation workflow");
        const header = node("div"); header.className = "hosted-canvas-header";
        const back = addButton(header, "hostedBack", boot.user.role === "admin" ? "← Admin workspace" : "← Workspace", () => leave(false, `/?dataset=${encodeURIComponent(dataset.id)}`));
        back.title = boot.user.role === "admin" ? "Save and return to datasets, quality analysis, compute, people and settings" : "Save and return to your assigned work";
        const toggle = node("button", "−"); toggle.type = "button"; toggle.id = "hostedPanelToggle"; toggle.setAttribute("aria-controls", "hostedWorkflowControls");
        toggle.setAttribute("aria-expanded", "true"); toggle.setAttribute("aria-label", "Minimize annotation controls"); header.append(toggle);
        const title = node("div"); title.className = "hosted-canvas-title"; title.append(node("strong", record.filename), node("small", `${dataset.name} · ${record.state === "confirmed" ? "Confirmed" : record.state === "submitted" ? "Submitted for review" : "Incoming"} · ${boot.user.username}`));
        const statusPanel = node("div"); statusPanel.className = "hosted-canvas-status";
        const state = node("span", "Loading…"); state.id = "hostedSaveState"; state.setAttribute("role", "status"); statusPanel.append(state);
        const content = node("div"); content.id = "hostedWorkflowControls";
        toggle.onclick = () => { content.hidden = !content.hidden; toggle.textContent = content.hidden ? "+" : "−"; toggle.setAttribute("aria-expanded", String(!content.hidden)); toggle.setAttribute("aria-label", content.hidden ? "Show annotation controls" : "Minimize annotation controls"); };
        const notes = node("div"); notes.id = "hostedReviewNotes"; content.append(notes);
        const bar = node("div"); bar.className = "hosted-canvas-bar"; content.append(bar);
        addButton(bar, "hostedSave", "Save now", () => save({ manual: true }));
        addButton(bar, "hostedReconnect", "Reconnect editing", () => acquire());
        const check = (id, label, checked) => { const wrap = node("label"), input = node("input"); input.type = "checkbox"; input.id = id; input.checked = checked; wrap.append(input, document.createTextNode(label)); bar.append(wrap); };
        check("hostedComplete", "Annotations checked", record.complete);
        check("hostedNegative", "No objects to label", record.negative);
        if (boot.user.role === "annotator") addButton(bar, "hostedSubmit", "Submit & next", () => transition("submit"));
        else if (["unassigned", "submitted"].includes(record.state)) addButton(bar, "hostedApprove", "Approve & next", () => transition("approve"));
        const feedbackPanel = node("div"); feedbackPanel.className = "hosted-canvas-feedback";
        if (boot.user.role === "admin" && !["assigned", "returned"].includes(record.state)) {
            addButton(feedbackPanel, "hostedArchive", "Archive image…", async () => {
                if (confirm(`Archive “${record.filename}”? It will leave the active dataset and work lists. The image, annotations and history stay recoverable from Export & recovery.`)) await transition("archive");
            });
        }
        if (boot.user.role === "admin" && record.state === "submitted") {
            const feedback = node("textarea"); feedback.id = "hostedFeedback"; feedback.rows = 2; feedback.placeholder = "What needs changing?"; feedback.setAttribute("aria-label", "Feedback for the annotator"); feedbackPanel.append(feedback);
            addButton(feedbackPanel, "hostedReturn", "Return for changes", () => transition("return"));
        }
        const takeover = addButton(feedbackPanel, "hostedTakeover", "Take over my other editing tab", async () => { if (confirm("Stop editing in your other tab and continue here? Unsent changes there will not overwrite this tab.")) await acquire(true); });
        takeover.title = "Only your own editing tab can be replaced this way.";
        if (boot.user.role === "admin" && ["assigned", "returned"].includes(record.state)) {
            takeover.hidden = true;
            addButton(feedbackPanel, "hostedOverride", "Recover from annotator", async () => {
                if (!confirm("Remove this image from the annotator's work? Their saved annotations are preserved; their editing access ends immediately.")) return;
                await api(`${base}/override`, { ...context(), feedback: "Administrator recovered the image from the assignment." }); location.reload();
            });
        }
        addButton(feedbackPanel, "hostedClassRequest", "Request a class", () => $("hostedClassDialog").showModal());
        const dialog = node("dialog"); dialog.id = "hostedClassDialog";
        const form = node("form"), name = node("input"), reason = node("textarea"); name.required = true; name.placeholder = "Class name"; name.setAttribute("aria-label", "Requested class name"); reason.required = true; reason.placeholder = "Why is this class needed?"; reason.setAttribute("aria-label", "Reason for class request");
        form.append(node("h2", "Request a dataset class"), name, reason);
        addButton(form, "hostedSendClass", "Send request", async () => { if (!form.reportValidity()) return; await api(`/api/hosted/datasets/${dataset.id}/class-requests`, { name: name.value, reason: reason.value, operation_id: uuid() }); dialog.close(); status("Class request sent. Existing classes remain available."); });
        addButton(form, "hostedCancelClass", "Cancel", () => dialog.close()); form.onsubmit = (e) => e.preventDefault(); dialog.append(form);
        content.append(feedbackPanel); panel.append(header, title, statusPanel, content);
        document.body.append(panel, dialog);
        window.dispatchEvent(new Event("tator:hosted-canvas"));
        update();
    }
    async function start() {
        try {
            const sessionCheck = verifySession();
            await sessionCheck;
            const retained = sessionStorage.getItem(key);
            const recovery = retained ? JSON.parse(retained) : null;
            if (recovery?.state === "committed_restricted") throw new Error("Your earlier change was saved, but its result is no longer available to this account. A later draft is retained in this tab. Ask an administrator to help recover it.");
            let recoveredReceipt = recovery?.pending ? await commandReceipt(recovery.pending) : null;
            if (recoveredReceipt && !recovery.pending.path.endsWith("/save")) {
                clearJournal(); location.assign("/"); return;
            }
            record = await api(`${base}?include_review=true`); dataset = await api(`/api/hosted/datasets/${record.dataset_id}`);
            buildUi();
            try { await acquire(false, false, record, sessionCheck); } catch (error) { report(error); }
            const neighbors = [];
            const filter = boot.user.role === "admin" ? `&state=${encodeURIComponent(record.state)}` : "";
            const page = offset => api(`/api/hosted/images?dataset_id=${dataset.id}&limit=500&compact=true&offset=${offset}${filter}`);
            const firstPage = await page(0); neighbors.push(...firstPage.images);
            if (firstPage.images.length === 500) {
                for (let offset = 500; ; offset += 2000) {
                    // Bounded parallel reads retain server order and the full
                    // work list without one network round trip per page.
                    const pages = await Promise.all([0, 500, 1000, 1500].map(step => page(offset + step)));
                    for (const result of pages) neighbors.push(...result.images);
                    if (pages.some(result => result.images.length < 500)) break;
                }
            }
            await window.TatorHostedCanvas.open(record, dataset, !editable, neighbors);
            const start = Date.now();
            while (!window.TatorHostedCanvas.ready()) { if (Date.now() - start > 30000) throw new Error("Image loading did not finish. Reopen it to retry."); await new Promise((resolve) => setTimeout(resolve, 16)); }
            ready = true; lastSaved = JSON.stringify(draft()); window.TatorHostedCanvas.freeze(!editable); update();
            if (retained) {
                recoveryRequired = true;
                freeze();
                status("A draft from the earlier connection is retained in this tab. Recover it before continuing.", true);
                addButton($("hostedSaveState").parentElement, "hostedResolve", "Recover my draft", async () => {
                    let receipt = recovery.pending ? await commandReceipt(recovery.pending) : null;
                    if (receipt && !recovery.pending.path.endsWith("/save")) { clearJournal(); location.assign("/"); return; }
                    let rejectedByLease = false;
                    if (recovery.pending && !receipt) {
                        // A missing receipt alone does not resolve a lost reply.
                        // Replay the original command before considering a new
                        // editing reservation or a different operation ID.
                        const original = recovery.pending;
                        try { receipt = {result: checkedResult(original, await api(original.path, original.body))}; }
                        catch (error) {
                            const certified = error.status === 422 && error.commit_outcome === "rejected" && error.operation_id === original.body.operation_id;
                            rejectedByLease = [409, 423].includes(error.status) && ["editor_lease_expired", "editor_lease_conflict", "ownership_changed", "annotation_version_changed"].includes(error.code);
                            if (!certified && !rejectedByLease) throw error;
                        }
                    }
                    const fresh = await api(base);
                    const anchor = recovery.pending?.body || recovery.unsent?.body || recovery.rejected?.context;
                    const expected = receipt?.result.current_version || anchor?.expected_version;
                    if (fresh.current_version !== expected) throw new Error("Someone saved a newer version. Your earlier draft is retained; it cannot replace their changes automatically. Ask the administrator to reconcile it.");
                    const restored = recovery.latest || recovery.pending?.body || recovery.unsent?.body;
                    if (!restored.annotations) throw new Error("No unsent annotations are attached to this operation. Check the workspace for its current review status.");
                    if (!confirm("Restore your retained draft on this unchanged server version? This also takes over your own previous editing tab.")) return;
                    record = fresh; await acquire(true, true);
                    if (rejectedByLease && record.generation <= anchor.generation) throw new Error("The earlier editing reservation has not been replaced. Your original command and draft remain retained; reconnect before trying again.");
                    window.TatorHostedCanvas.restoreDraft(restored.annotations);
                    $("hostedComplete").checked = restored.complete;
                    $("hostedNegative").checked = restored.negative;
                    // Restoration is complete even if this draft still needs
                    // correction. Subsequent edits must use the live canvas,
                    // not the old recovery journal, as their source of truth.
                    recoveryRequired = false; $("hostedResolve").remove();
                    pending = null; await save({ manual: true });
                });
            } else if (editable) status("Connected · edits save automatically");
            await refreshFeedback();
        } catch (error) { if (!$("hostedSaveState")) { document.body.replaceChildren(node("p", error.message)); const back = node("a", "Return to sign in"); back.href = "/"; document.body.append(back); } else report(error); }
    }
    async function applyVersionedProposal(path, fields, expectedVersion, message, eventName, retryId) {
        if (!window.TatorHosted.editable) throw new Error("Reconnect editing before using this proposal.");
        const origin = {imageId, base, key};
        const checkOrigin = () => { if (imageId !== origin.imageId || base !== origin.base || key !== origin.key || accountChanged) throw new Error("The active image or account changed. Reopen the original image to check its saved proposal."); };
        busy = true; update();
        try {
        if (!await save({manual: true})) throw new Error("Save this image before using the proposal.");
        checkOrigin();
        if (expectedVersion !== record.current_version) throw new Error("This image changed after the comparison was prepared. Reopen the comparison before choosing a version.");
        const operation = {path, body: {...context(), ...fields}};
        pending = operation; journal({pending: operation, latest: draft()}); handoff = true; update();
        const finish = async () => {
            checkOrigin();
            const fresh = await api(origin.base);
            checkOrigin(); record = fresh;
            window.TatorHostedCanvas.acknowledge(record);
            window.TatorHostedCanvas.restoreDraft(record.annotations);
            lastSaved = JSON.stringify(draft()); pending = null; clearJournal(); handoff = false;
            $(retryId)?.remove();
            await acquire(); status(message); update();
            window.dispatchEvent(new Event(eventName));
        };
        try { checkedResult(operation, await api(operation.path, operation.body)); await finish(); }
        catch (error) {
            if (!$(retryId)) addButton($("hostedSaveState").parentElement, retryId, "Check saved proposal result", async () => {
                const receipt = await commandReceipt(operation);
                if (receipt?.result_restricted === true) {
                    retainRestrictedCommit(operation, receipt);
                    throw new Error("The proposal was saved, but your account can no longer view its result. Return to the workspace to check your access.");
                }
                if (!receipt) checkedResult(operation, await api(operation.path, operation.body));
                await finish();
            });
            report(error); throw error;
        }
        } finally { busy = false; update(); }
    }
    window.TatorHosted = {
        get editable() { return editable && ready && !handoff && !busy && !recoveryRequired; },
        editorId, get imageId() { return imageId; }, instanceId: boot.instance_id, get mediaUrl() { return `${base}/media`; }, save, navigate,
        mediaUrlFor: (id) => `/api/hosted/images/${encodeURIComponent(id)}/media`,
        get datasetId() { return dataset?.id; }, api,
        async capturePaidBatchContext() {
            if (!window.TatorHosted.editable) throw new Error("Reconnect editing before preparing a paid caption batch.");
            busy = true; update();
            const origin = imageId;
            try {
                if (!await save({manual: true})) throw new Error("Save this image before preparing a caption batch.");
                if (imageId !== origin || accountChanged) throw new Error("The image or account changed. Prepare the batch again.");
                return {dataset_id: dataset.id, image_context: {image_id: imageId, expected_version: record.current_version, generation: record.generation, editor_id: editorId}};
            } finally { busy = false; update(); }
        },
        applyImportConflict(conflictId, {choice, negative, expectedVersion}) {
            if (boot.user.role !== "admin") throw new Error("An administrator must resolve imported annotation conflicts.");
            return applyVersionedProposal(`${base}/import-conflicts/${encodeURIComponent(conflictId)}/resolve`, {choice, negative}, expectedVersion,
                "Import comparison resolved · previous annotations preserved", "tator:import-conflict-resolved", "hostedImportConflictRetry");
        },
        applyInstructionReview(review, decisions) {
            if (boot.user.role !== "admin") throw new Error("An administrator must review training language.");
            if (review.image_id !== imageId) throw new Error("This review describes a different image.");
            return applyVersionedProposal(`/api/hosted/tools/images/${encodeURIComponent(imageId)}/instruction-review`,
                {source_sha256: review.source_sha256, decisions}, review.expected_version,
                "Language review saved · model evidence preserved", "tator:instruction-reviewed", "hostedInstructionReviewRetry");
        },
        applyPaidCaptionProposal(jobId, proposal, {includeGeneratedQa = true, replacePrimary = true} = {}) {
            if (proposal.image_id !== imageId) throw new Error("This proposal describes a different image.");
            return applyVersionedProposal(`/api/hosted/tools/paid-batches/${encodeURIComponent(jobId)}/apply`, {image_id: imageId, proposal_sha256: proposal.proposal_sha256,
                include_generated_qa: includeGeneratedQa, replace_primary: replacePrimary}, proposal.version_id,
                "Caption saved · earlier captions preserved", "tator:caption-applied", "hostedPaidCaptionRetry");
        },
        async applyCaptionProposal(batchId, proposal, {includeGeneratedQa = true, replacePrimary = true} = {}) {
            if (!window.TatorHosted.editable || boot.user.role !== "admin") throw new Error("Reconnect administrator editing before using this caption.");
            if (proposal.image_id !== imageId) throw new Error("This proposal describes a different image.");
            return applyVersionedProposal(`/api/hosted/tools/caption-batches/${encodeURIComponent(batchId)}/images/${encodeURIComponent(imageId)}/apply`,
                {proposal_sha256: proposal.proposal_sha256, include_generated_qa: includeGeneratedQa, replace_primary: replacePrimary}, proposal.source_version,
                "Caption saved · earlier captions preserved", "tator:caption-applied", "hostedCaptionRetry");
        },
        setToolsState(value) { toolState = value; },
        get samConfigured() { return Boolean(toolState?.capabilities?.sam_point?.configured); },
        get edrReady() { return Boolean(toolState?.capabilities?.edr_prepass?.configured) && toolState?.phase === "available"; },
        get automationPauseReason() { return toolState && toolState.phase !== "available" ? "Annotation assistance is paused while server compute changes. You can still annotate and save manually." : null; },
        async fetch(url, options = {}) {
            const target = new URL(url instanceof Request ? url.url : url, location.href);
            if (target.origin !== location.origin) throw new Error("Hosted tools use this server only.");
            if (target.pathname.startsWith("/qwen/caption/openai_batches") && window.TatorHostedPaidBatches) return window.TatorHostedPaidBatches.legacyFetch(target, options);
            if (target.pathname === "/qwen/caption/openai_metadata") {
                const catalog = await api(`/api/hosted/tools/catalog?dataset_id=${encodeURIComponent(dataset.id)}`);
                return jsonResponse({models: (catalog.paid_caption?.models || []).map(model => ({...model, supports_original_image_detail: model.image_details.includes("original")}))});
            }
            const qwenMeta = ["/qwen/status", "/qwen/models", "/qwen/progress", "/qwen/cancel", "/qwen/caption/cancel"].includes(target.pathname);
            const installedMeta = Boolean(metadataRoutes[target.pathname]) || target.pathname === "/sam3/models/activate" || target.pathname.startsWith("/agent_mining/cascades") || target.pathname.startsWith("/agent_mining/recipes/") || target.pathname.startsWith("/prepass/recipes/");
            if (qwenMeta || installedMeta || toolRoutes[target.pathname] || target.pathname.startsWith("/qwen/caption/jobs")) {
                try {
                    if (qwenMeta) return await qwenMetadata(target.pathname);
                    if (installedMeta) return await installedToolsFetch(target, options);
                    return target.pathname.startsWith("/qwen/caption/jobs") ? await captionJobFetch(target, options) : await toolFetch(toolRoutes[target.pathname], options);
                }
                catch (error) {
                    if (error.name === "AbortError") throw error;
                    return new Response(JSON.stringify({ detail: error.message, error: error.code || "hosted_tool_failed" }), { status: error.status || 409, headers: { "content-type": "application/json" } });
                }
            }
            return nativeFetch(url, { ...options, headers: headers(options), credentials: "same-origin" });
        },
    };
    window.addEventListener("load", start, { once: true });
    setInterval(() => {
        if (editable && ready && !busy && !handoff) {
            if (saving && pending) { try { journal({ pending, latest: draft() }); } catch (error) { report(error); } }
            else save().catch(() => {});
        }
    }, 2000);
    setInterval(async () => {
        if (!record || !editable || busy || saving || handoff) return;
        try { await verifySession(); await api(`${base}/editor`, { editor_id: editorId, expected_version: record.current_version, generation: record.generation }); }
        catch (error) { report(error); }
    }, 30000);
    window.addEventListener("beforeunload", (event) => {
        // Undo may reach the saved baseline before the next autosave tick.
        // Reconcile that transition even when there is nothing left to send.
        try { retainDraft(); } catch (error) { status(error.message, true); }
        if (storageFailed || pending || (ready && JSON.stringify(draft()) !== lastSaved)) {
            event.preventDefault(); event.returnValue = "";
        }
    });
    window.addEventListener("pagehide", () => {
        if (!editable || !record || handoff || pending || saving) return;
        nativeFetch(`${base}/release`, { method: "POST", headers: { ...Object.fromEntries(headers()), "content-type": "application/json" }, body: JSON.stringify(context()), credentials: "same-origin", keepalive: true }).catch(() => {});
    });
})();
