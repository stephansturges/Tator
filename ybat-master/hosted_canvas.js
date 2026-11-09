/* Hosted bridge for the existing first-pane canvas. Authority stays on the server. */
(() => {
    "use strict";
    const boot = window.TatorHostedBoot;
    if (!boot) return;
    // Lay the hosted page out at the device's real width, so phones get the
    // "larger screen" notice instead of a shrunken desktop layout.
    if (!document.querySelector('meta[name="viewport"]')) {
        const viewport = document.createElement("meta"); viewport.name = "viewport"; viewport.content = "width=device-width, initial-scale=1";
        document.head.append(viewport);
    }
    const nativeFetch = window.fetch.bind(window);
    const editorId = crypto.randomUUID();
    let accountChanged = false;
    let imageId = boot.image_id;
    let base = `/api/hosted/images/${encodeURIComponent(imageId)}`;
    let key = `tator.hosted.draft:${boot.instance_id}:${boot.user.user_id}:${imageId}`;
    let record, dataset, editable = false, ready = false, pending = null, saving = null, lastSaved = "", busy = false, handoff = false, recoveryRequired = false, toolState = null, retentionCommand = null;
    let rejected = null, storageFailed = false, ownLeaseConflict = false;
    let queue = null, tranche = boot.tranche || null;
    let scopeLeft = Boolean(boot.scope_left), originScope = boot.requested_scope || boot.scope;
    const sameScope = (a, b) => a?.dataset_id === b?.dataset_id && (a?.tranche_id || null) === (b?.tranche_id || null) && a?.queue === b?.queue;
    // Back returns to the workspace view, search, filter and annotator the
    // user came from, and asks it to scroll to the image they were last on.
    const pageContext = window.HostedQueue.pageContext();
    const scopeBackUrl = scope => window.HostedQueue.workspaceUrl(scope, pageContext, imageId);
    const backUrl = () => queue ? queue.back(imageId) : scopeBackUrl(originScope);
    async function openScope(scoped) {
        const nextQueue = new window.HostedQueue({api, scope: scoped.scope, context: pageContext, filterByAnnotator: boot.user.role === "admin",
            onAppend: (rows, options) => window.TatorHostedCanvas.appendQueue(rows, options), onReconcile: rows => window.TatorHostedCanvas.reconcileQueue(rows)});
        const rows = await nextQueue.open(imageId);
        queue = nextQueue; scopeLeft = false; updateTranche(scoped.tranche);
        return rows;
    }
    let mediaEpoch = 0, mediaCacheAllowed = false, mediaWarmTimer = null;
    const $ = (id) => document.getElementById(id);
    const mediaPrefetch = window.HostedMediaPrefetch ? new window.HostedMediaPrefetch({
        request: async (id, signal) => {
            if (accountChanged) throw new Error("The account changed.");
            const response = await nativeFetch(`/api/hosted/images/${encodeURIComponent(id)}/media`, {
                credentials: "same-origin", headers: headers(), cache: "no-store", priority: "low", signal,
            });
            if ([401, 403, 409].includes(response.status)) clearMediaPrefetch();
            return response;
        },
    }) : null;
    function pauseMediaPrefetch(keepId = null) {
        clearTimeout(mediaWarmTimer); mediaWarmTimer = null;
        mediaPrefetch?.pause(keepId);
    }
    function clearMediaPrefetch() {
        mediaEpoch++; mediaCacheAllowed = false;
        pauseMediaPrefetch(); mediaPrefetch?.clear();
    }
    function scheduleMediaPrefetch() {
        clearTimeout(mediaWarmTimer);
        // Let the current canvas paint first. No speculative decoding, labels,
        // editing reservations or durable browser storage are involved.
        mediaWarmTimer = setTimeout(async () => {
            if (!ready || busy || handoff || recoveryRequired || accountChanged || !mediaCacheAllowed || document.hidden) return;
            try { await queue?.ahead(imageId, 2); } catch (_) { /* Foreground navigation will expose a queue error. */ }
            if (ready && !busy && !handoff && !accountChanged && mediaCacheAllowed && !document.hidden) mediaPrefetch?.schedule(window.TatorHostedCanvas?.nextImageIds(2) || []);
        }, 150);
    }
    async function mediaBlobFor(id) {
        if (accountChanged) throw new Error("Your account changed. Reopen the workspace.");
        const epoch = mediaEpoch;
        const current = id === imageId && record?.id === id;
        const sourceHash = current ? record.media_hash : null;
        const check = () => {
            if (accountChanged || epoch !== mediaEpoch || (current && id !== imageId)) {
                throw new DOMException("The image or account changed.", "AbortError");
            }
        };
        if (current && mediaCacheAllowed) {
            const blob = await mediaPrefetch?.take(id, sourceHash);
            check();
            if (blob) return blob;
        }
        const response = await nativeFetch(`/api/hosted/images/${encodeURIComponent(id)}/media`, {
            credentials: "same-origin", headers: headers(), cache: "no-store", priority: current ? "high" : "auto",
        });
        check();
        if (!response.ok) {
            if (current || [401, 403, 409].includes(response.status)) clearMediaPrefetch();
            throw new Error(`Could not load this image (${response.status}). Reopen it to check your access.`);
        }
        const responseHash = response.headers.get("X-Tator-Source-Media-Hash");
        if (sourceHash && responseHash && sourceHash !== responseHash) {
            await response.body?.cancel();
            clearMediaPrefetch();
            throw new Error("The image source changed. Reopen it to load its current pixels and annotations.");
        }
        const blob = await response.blob();
        check();
        return blob;
    }
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
    const mac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
    const keyName = (key) => mac ? `⌘${key}` : `Ctrl+${key}`;
    const messages = {
        image_left_queue: "This image left the current tranche or queue. Your draft is retained. Use Back to tranche to continue.",
        queue_scope_changed: "This tranche changed. Your draft is retained. Return to the tranche to see its current work.",
        queue_scope_mismatch: "This image is no longer in the selected queue. Return to the tranche to continue.",
        annotation_version_changed: "This image changed on the server. Your unsaved draft is retained. Reload to compare before saving.",
        ownership_changed: "Editing ownership changed. Your draft is retained, but it cannot overwrite the new owner's work.",
        editor_lease_conflict: "Someone else is editing this image right now. You can view it; editing becomes available when they leave it.",
        editor_lease_expired: "Your editing connection expired. Your draft is retained. Reconnect before continuing.",
        login_required: "Your session ended. Sign in again; your unsent draft is retained for your account.",
        csrf_rejected: "Your account session changed in another tab. Reconnect before continuing.",
        installation_changed: "The server was restored or replaced. Reconnect before editing.",
        image_owned_by_annotator: "The assigned annotator is editing this image. You can view it, or explicitly recover it from their assignment.",
        negative_has_objects: "This image has boxes. Clear the empty-image check (“No objects to label”) or delete the boxes to save.",
        empty_annotation_requires_negative_review: "This image has no boxes. Tick “No objects to label” if there is nothing to label here, or draw the objects, then submit again.",
        annotation_not_complete: "Tick “Annotations checked” once you have reviewed every box, then submit again.",
        confirmed_annotation_requires_completion: "Confirmed images stay checked. Tick “Annotations checked” to save this change.",
        image_not_submittable: "This image can no longer be submitted from here. Your work is saved; return to the workspace to see its current state.",
        image_not_approvable: "This image can no longer be approved from here. Return to the workspace to see its current state.",
        image_not_returnable: "This image can no longer be returned from here. Return to the workspace to see its current state.",
        image_not_editable: "Your account can no longer edit this image. Your draft is retained in this tab.",
        image_not_found: "This image is no longer available to your account. Your draft is retained in this tab.",
        instruction_review_requires_admin: "Only an administrator can change review decisions. Your draft is retained.",
        machine_validation_requires_server_evidence: "These QA validation labels do not match a saved server result. Your draft is retained; ask an administrator to review the affected answers.",
        machine_validation_source_changed: "The caption result belongs to an earlier image version or editing session. Your draft is retained; generate it again from the current saved image.",
    };
    function status(text, error = false) { if ($("hostedSaveState")) { $("hostedSaveState").textContent = text; $("hostedSaveState").dataset.error = String(error); } }
    function freeze() {
        clearMediaPrefetch();
        // Editing authority and local recovery are separate. Capture the newest
        // draft while the original image/account context is still available.
        let retentionFailure = null;
        try { retainDraft(); } catch (error) { retentionFailure = error; }
        editable = false; window.TatorHostedCanvas?.freeze(); update();
        return retentionFailure;
    }
    // Content refusals ask the user to correct the image; they never end editing.
    const contentRejections = new Set(["empty_annotation_requires_negative_review", "annotation_not_complete", "confirmed_annotation_requires_completion", "negative_has_objects"]);
    const offlineMessage = "The connection to the server was lost. Your work is kept in this tab and saves automatically when the connection returns.";
    function report(error) {
        if (pending && error.commit_outcome === "committed" && error.result_restricted === true && error.operation_id === pending.body.operation_id) {
            retainRestrictedCommit(pending, error);
        }
        const retentionFailure = [401, 403, 409, 423].includes(error.status) && !contentRejections.has(error.code) ? freeze() : null;
        status(retentionFailure?.message || error.message || String(error), true);
        if (error.status === 401 && !retentionFailure && !storageFailed) offerSignIn();
        if (error.code === "ownership_changed") explainOwnership();
    }
    // An expired session keeps the draft in this tab; signing in again in the
    // same tab and reopening the image offers to recover it.
    let leavingToSignIn = false;
    function offerSignIn() {
        const parent = $("hostedSaveState")?.parentElement;
        if (!parent || $("hostedSignIn")) return;
        const button = node("button", "Sign in again"); button.type = "button"; button.id = "hostedSignIn";
        button.title = "Your unsent draft stays in this tab. After signing in, reopen this image to recover it.";
        button.onclick = () => { leavingToSignIn = true; location.assign(backUrl()); };
        parent.append(button);
    }
    async function explainOwnership() {
        // Taking the image over in another tab of yours ends editing here.
        const origin = imageId;
        try {
            const current = await api(base);
            if (imageId === origin && !editable && current.editor_lease?.actor_id === boot.user.user_id) {
                status("You continued this image in another tab or window, so editing stopped here. Changes made here since then stay in this tab only; close it, or reload it to see the saved version.", true);
            }
        } catch (_) { /* Keep the original message. */ }
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
        let response;
        try { response = await nativeFetch(path, options); }
        catch (error) {
            // A network failure says nothing about the server's state.
            if (error?.name === "AbortError") throw error;
            throw Object.assign(new Error(offlineMessage), {offline: true});
        }
        let value;
        try { value = await response.json(); }
        catch (_) { throw new Error("The server reply was incomplete. Your pending change is retained; reconnect or save again to check its original result."); }
        if (accountChanged) throw Object.assign(new Error("Your account changed while this request was running. Its result is retained for the original account."), {status: 401});
        if (!response.ok) {
            const fallback = value?.error ? `The server did not accept this request (${value.error}). Your work is kept in this tab.` : `Request failed (${response.status}).`;
            const error = new Error(messages[value?.error] || (typeof value?.detail === "string" ? value.detail : fallback)); error.status = response.status; error.code = value?.error;
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
        clearMediaPrefetch();
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
        try { sessionStorage.setItem(key, JSON.stringify({ schema: 2, scope: queue?.scope || originScope, state: value.pending ? "pending" : value.rejected ? "rejected" : "unsent", ...value })); storageFailed = false; }
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
            if ($(id)) $(id).disabled = !editable || !ready || busy || handoff || recoveryRequired;
        }
        for (const id of ["hostedRefreshQueue", "hostedMoreImages"]) if ($(id)) $(id).disabled = !queue || scopeLeft || recoveryRequired || busy;
        if ($("hostedReconnect")) $("hostedReconnect").hidden = editable || scopeLeft || recoveryRequired;
        // The server only lets you replace your own editing tab; never offer
        // a takeover that would fail because another person holds the image.
        if ($("hostedTakeover")) $("hostedTakeover").hidden = editable || recoveryRequired || !ownLeaseConflict || (boot.user.role === "admin" && ["assigned", "returned"].includes(record.state));
        syncEditButtons();
    }
    const userNames = new Map();
    async function personName(userId) {
        if (!userId) return null;
        if (userId === boot.user.user_id) return "you";
        const row = queue?.rows.find(item => item.assignee_id === userId && item.assignee_name);
        if (row) return row.assignee_name;
        if (boot.user.role !== "admin") return null;
        if (!userNames.size) {
            try { for (const user of (await api("/api/hosted/users")).users || []) userNames.set(user.id, user.display_name || user.username); }
            catch (_) { return null; }
        }
        return userNames.get(userId) || null;
    }
    async function explainLeaseConflict(error, current) {
        // The image record names the lease holder; the 423 reply does not.
        const holder = current?.editor_lease?.actor_id;
        ownLeaseConflict = holder === boot.user.user_id;
        if (ownLeaseConflict) {
            error.message = "This image is open in another editing tab or window of yours, possibly one you closed in the last two minutes. Taking over here is safe: unsent changes there will not overwrite this tab.";
        } else {
            const name = await personName(holder);
            error.message = `${name ? name.charAt(0).toUpperCase() + name.slice(1) : "Someone else"} is editing this image right now. You can view it; editing becomes available when they leave it.`;
        }
        return error;
    }
    async function save({ manual = false } = {}) {
        if (saving) { await saving; if (manual) return save({ manual }); return false; }
        if (!ready || !editable || handoff || recoveryRequired) return false;
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
    async function run(button, action) { if (busy) return; busy = true; pauseMediaPrefetch(); update(); try { await action(); } catch (error) { report(error); } finally { busy = false; update(); scheduleMediaPrefetch(); } }
    // Undo and redo for the open image's objects. Each finished edit (a
    // release, a key, a class choice or a tool result) is one step; undoing
    // restores the objects on the canvas, and autosave stores the result.
    const undo = { image: null, past: [], future: [] };
    let pointerHeld = false;
    function objectState() { return JSON.stringify(window.TatorHostedCanvas.snapshot().objects || []); }
    function resetUndo() {
        undo.image = imageId; undo.future = [];
        try { undo.past = [objectState()]; } catch (_) { undo.past = []; }
        syncEditButtons();
    }
    function noteEdit() {
        if (!ready || recoveryRequired || pointerHeld || undo.image !== imageId || window.TatorHostedCanvas.hasPendingTools()) return;
        let current;
        try { current = objectState(); } catch (_) { return; }
        if (undo.past.at(-1) === current) return;
        undo.past.push(current); if (undo.past.length > 100) undo.past.shift();
        undo.future = []; syncEditButtons();
    }
    function stepUndo(direction) {
        if (!window.TatorHosted.editable) { status("Undo needs editing access. Reconnect editing first.", true); return; }
        if (window.TatorHostedCanvas.hasPendingTools()) { status("Wait for the annotation tool to finish before undoing.", true); return; }
        noteEdit();
        if (direction < 0) {
            if (undo.past.length < 2) { status("Nothing to undo on this image."); return; }
            undo.future.push(undo.past.pop());
        } else {
            if (!undo.future.length) { status("Nothing to redo on this image."); return; }
            undo.past.push(undo.future.pop());
        }
        const annotations = window.TatorHostedCanvas.snapshot();
        annotations.objects = JSON.parse(undo.past.at(-1));
        window.TatorHostedCanvas.restoreDraft(annotations);
        syncEditButtons();
        status(`${direction < 0 ? "Undone" : "Redone"} · saves automatically`);
    }
    function deleteSelection() {
        if (!window.TatorHosted.editable) { status("Deleting needs editing access. Reconnect editing first.", true); return; }
        if (!window.TatorHostedCanvas.deleteSelection()) { status("Click or tap a box first, then delete it."); return; }
        noteEdit(); status(`Box deleted · ${keyName("Z")} undoes it`);
    }
    function syncEditButtons() {
        const blocked = !editable || !ready || busy || handoff || recoveryRequired;
        if ($("hostedUndo")) $("hostedUndo").disabled = blocked || undo.past.length < 2;
        if ($("hostedRedo")) $("hostedRedo").disabled = blocked || !undo.future.length;
        if ($("hostedDeleteBox")) $("hostedDeleteBox").disabled = blocked;
    }
    for (const type of ["pointerup", "keyup", "change", "touchend"]) document.addEventListener(type, () => { if (type !== "keyup") pointerHeld = false; setTimeout(noteEdit, 0); }, true);
    document.addEventListener("pointerdown", event => { pointerHeld = event.target?.id === "canvas"; }, true);
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
        let lease;
        try { lease = await api(`${base}/editor`, { editor_id: editorId, expected_version: fresh.current_version, generation: fresh.generation, takeover }); }
        catch (error) {
            if (error.code === "editor_lease_conflict") {
                // The reply does not say who holds the image; read it again.
                let holder = fresh;
                if (preparedRecord) try { holder = await api(base); } catch (_) { /* Keep the earlier record. */ }
                await explainLeaseConflict(error, holder); update();
            }
            throw error;
        }
        ownLeaseConflict = false;
        record = { ...fresh, ...lease }; editable = true;
        mediaCacheAllowed = true;
        window.TatorHostedCanvas?.freeze(false); status("Connected · edits save automatically"); update();
    }
    async function leave(next = false, destination = backUrl()) {
        pauseMediaPrefetch();
        if (editable && !handoff) { if (!await save({ manual: true })) throw new Error("Save your work before leaving."); await api(`${base}/release`, context()); editable = false; }
        if (next && queue) {
            let image = await queue.neighbor(imageId, 1);
            while (image) {
                try { await queue.authorize(image.id); location.assign(queue.url(image.id)); return; }
                catch (error) {
                    if (![403, 404, 409].includes(error.status)) throw error;
                    queue.remove(image.id); window.TatorHostedCanvas.removeQueue([image.id]);
                    image = await queue.neighbor(imageId, 1);
                }
            }
            // Nothing left here: say so briefly, then return to the same view.
            status(queue.scope.tranche_id ? "Tranche finished · returning to the workspace…" : "Queue finished · returning to the workspace…");
            const target = queue.back(imageId);
            await new Promise(resolve => setTimeout(resolve, 900));
            location.assign(target); return;
        }
        location.assign(destination);
    }
    function updateTranche(value) {
        tranche = value;
        if ($("hostedBreadcrumb")) $("hostedBreadcrumb").textContent = tranche ? `${dataset.name} > ${tranche.name}` : dataset.name;
        if ($("hostedBack")) $("hostedBack").textContent = backLabel();
        const title = $("hostedTrancheTitle"), guidance = $("hostedTrancheInstructions");
        if (title) { title.textContent = tranche ? `${tranche.name} · ${tranche.lifecycle === "draft" ? `${tranche.counts.not_assigned} in Draft` : `${tranche.counts.to_finish + tranche.counts.changes_requested} to finish`} · ${tranche.counts.awaiting_review} awaiting review` : ""; title.hidden = !tranche; }
        if (guidance) { guidance.textContent = tranche?.instructions || ""; guidance.hidden = !tranche?.instructions; }
    }
    async function step(delta = 1) {
        // Next/Previous pressed while the list is loading or a save finishes
        // waits briefly for it instead of being silently dropped.
        for (let waited = 0; busy && !handoff && waited < 5000; waited += 50) await new Promise(resolve => setTimeout(resolve, 50));
        if (!queue || busy || handoff) return;
        try { const target = await queue.neighbor(imageId, delta); if (target) await navigate(target.id); else status(delta > 0 ? "You reached the end of this queue." : "You reached the beginning of this queue."); }
        catch (error) { report(error); }
    }
    async function navigate(targetId) {
        if (!queue || scopeLeft || recoveryRequired) throw new Error("Recover the retained draft or open its current work queue before changing images.");
        if (busy || handoff) throw new Error("Wait for the current save or handoff before changing images.");
        if (targetId === imageId) return;
        if ([...activeTools.values()].some((item) => !["sam_preload", "sam_state"].includes(item.operation))) throw new Error("Wait for the current annotation tool, or cancel it before changing images.");
        busy = true; pauseMediaPrefetch(targetId); update();
        try {
            const sessionCheck = verifySession();
            const [, target, scoped] = await Promise.all([sessionCheck, api(`/api/hosted/images/${encodeURIComponent(targetId)}?include_review=true`), queue.authorize(targetId)]);
            updateTranche(scoped.tranche);
            if (target.dataset_id !== dataset.id) throw new Error("Open the other dataset from the workspace.");
            if (editable) {
                if (!await save({ manual: true })) throw new Error("Save your current image before changing images.");
                await api(`${base}/release`, context()); editable = false;
            }
            const targetKey = `tator.hosted.draft:${boot.instance_id}:${boot.user.user_id}:${targetId}`;
            if (sessionStorage.getItem(targetKey) || target.state !== record.state) { location.assign(queue.url(targetId)); return; }
            ready = false; editable = false; pending = null; rejected = null; retentionCommand = null;
            imageId = targetId; base = `/api/hosted/images/${encodeURIComponent(imageId)}`; key = targetKey; record = target;
            let accessError = null;
            try { await acquire(false, false, target, sessionCheck); } catch (error) { accessError = error; editable = false; clearMediaPrefetch(); }
            history.replaceState(null, "", queue.url(imageId));
            $("hostedComplete").checked = record.complete; $("hostedNegative").checked = record.negative;
            document.querySelector(".hosted-canvas-title strong").textContent = record.filename;
            if ($("hostedStateLine")) $("hostedStateLine").textContent = `${dataset.name} · ${stateText()} · ${boot.user.username}`;
            document.querySelectorAll(".hosted-review-note").forEach((item) => item.remove());
            window.TatorHostedCanvas.activate(record, dataset);
            const began = Date.now();
            while (!window.TatorHostedCanvas.ready()) {
                if (Date.now() - began > 30000) throw new Error("This image did not finish loading. Reopen it to retry.");
                await new Promise((resolve) => setTimeout(resolve, 16));
            }
            ready = true; lastSaved = JSON.stringify(draft()); resetUndo();
            window.TatorHostedCanvas.freeze(!editable);
            window.dispatchEvent(new Event("tator:hosted-image"));
            history.replaceState(null, "", queue.url(imageId));
            await refreshFeedback();
            if (accessError) report(accessError);
            else status("Connected · edits save automatically");
        } catch (error) {
            if (targetId !== imageId && ["image_left_queue", "queue_scope_mismatch"].includes(error.code)) {
                queue.remove(targetId); window.TatorHostedCanvas.removeQueue([targetId]);
                status("That image left this queue. Your current image and draft are unchanged.", true);
            } else { if ([401, 403, 404, 409, 423].includes(error.status)) clearMediaPrefetch(); report(error); }
            throw error;
        }
        finally { busy = false; update(); scheduleMediaPrefetch(); }
    }
    async function transition(kind) {
        if (window.TatorHostedCanvas.hasPendingTools() || [...activeTools.values()].some((item) => !["sam_preload", "sam_state"].includes(item.operation))) throw new Error("Wait for the annotation tool to finish, or cancel it before handing over this image.");
        if (pending && !pending.path.endsWith("/save")) throw new Error("Check the earlier handoff before submitting again.");
        if (!["return", "archive"].includes(kind)) {
            // Check what the server will require before handing the image over,
            // so a refusal never strands the image mid-handoff.
            const objects = window.TatorHostedCanvas.snapshot().objects || [];
            const negative = $("hostedNegative").checked;
            if (!objects.length && !negative) throw new Error(messages.empty_annotation_requires_negative_review);
            if (objects.length && negative) throw new Error(messages.negative_has_objects);
            // Confirming an empty image is itself the check.
            if (negative && !$("hostedComplete").checked) $("hostedComplete").checked = true;
            if (!$("hostedComplete").checked) throw new Error(messages.annotation_not_complete);
        }
        if (!await save({ manual: true })) throw new Error("Save before handing over this image.");
        const body = context();
        if (kind === "return") { body.feedback = $("hostedFeedback").value.trim(); if (!body.feedback) throw new Error("Explain what needs changing before returning the image."); }
        if (kind === "archive") body.reason = "Archived by an administrator from the annotation canvas.";
        handoff = true; update();
        const operation = { path: `${base}/${kind}`, body };
        pending = operation;
        journal({ pending: operation, latest: draft() });
        const recoveryUrl = new URL(location.href); recoveryUrl.searchParams.set("recovery", operation.body.operation_id);
        history.replaceState(null, "", recoveryUrl.pathname + recoveryUrl.search);
        try {
            try { checkedResult(operation, await api(operation.path, operation.body)); }
            catch (error) { await settleRejectedTransition(operation, error); throw error; }
            pending = null; clearJournal(); editable = false; await leave(kind !== "archive");
        }
        catch (error) {
            // A transport failure says nothing about whether the server committed.
            // Retain the exact command until its receipt or replay resolves it.
            if (!error.rejectedTransition && !$("hostedHandoffRetry")) addButton($("hostedSaveState").parentElement, "hostedHandoffRetry", "Check handoff result", async () => {
                const receipt = await commandReceipt(operation);
                if (!receipt) {
                    try { checkedResult(operation, await api(operation.path, operation.body)); }
                    catch (error) { await settleRejectedTransition(operation, error); throw error; }
                }
                pending = null; clearJournal(); editable = false; await leave(kind !== "archive");
            });
            if (error.offline) error.message = "The connection was lost while handing over this image. Your work is kept; when the connection returns, use “Check handoff result”.";
            throw error;
        }
    }
    function definitiveRefusal(error) {
        // A Tator error code in a 4xx reply is the server refusing the command.
        return Boolean(error?.code) && error.status >= 400 && error.status < 500 && error.status !== 401 && !error.result_restricted && error.commit_outcome !== "committed";
    }
    async function settleRejectedTransition(operation, error) {
        // Only a refusal with no receipt for this exact operation proves it
        // never committed. Then the handoff is over: editing continues on the
        // saved draft, and recovery will not replay the refused command.
        if (!definitiveRefusal(error)) return false;
        let receipt;
        try { receipt = await commandReceipt(operation); } catch (_) { return false; }
        if (receipt !== null || pending !== operation) return false;
        pending = null; handoff = false; error.rejectedTransition = true;
        try { retainDraft(); } catch (retentionError) { status(retentionError.message, true); }
        const url = new URL(location.href); url.searchParams.delete("recovery"); history.replaceState(null, "", url.pathname + url.search);
        $("hostedHandoffRetry")?.remove();
        update();
        return true;
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
        const back = addButton(header, "hostedBack", backLabel(), () => leave(false, backUrl()));
        back.title = boot.user.role === "admin" ? "Save and return to the workspace view you came from" : "Save and return to your assigned work";
        const toggle = node("button", "−"); toggle.type = "button"; toggle.id = "hostedPanelToggle"; toggle.setAttribute("aria-controls", "hostedWorkflowControls");
        // The page's own theme button (light/dark, double-click for Pip-Boy) lives in the hidden tab bar; keep it reachable here.
        const headerTools = node("div"); headerTools.className = "hosted-canvas-header-tools";
        const themeButton = document.getElementById("themeToggleButton"); if (themeButton) headerTools.append(themeButton);
        headerTools.append(toggle); header.append(headerTools);
        const stateLine = node("small", `${dataset.name} · ${stateText()} · ${boot.user.username}`); stateLine.id = "hostedStateLine";
        const title = node("div"); title.className = "hosted-canvas-title"; title.append(node("strong", record.filename), stateLine);
        const breadcrumb = node("small"); breadcrumb.id = "hostedBreadcrumb"; title.append(breadcrumb);
        const trancheTitle = node("strong"); trancheTitle.id = "hostedTrancheTitle"; title.append(trancheTitle);
        const statusPanel = node("div"); statusPanel.className = "hosted-canvas-status";
        const state = node("span", "Loading…"); state.id = "hostedSaveState"; state.setAttribute("role", "status"); statusPanel.append(state);
        const content = node("div"); content.id = "hostedWorkflowControls";
        // Minimized is a small pill. The choice survives Submit & next, which
        // loads a new page, for the rest of this browser tab.
        const minimize = (value, remember = true) => {
            content.hidden = value; panel.classList.toggle("is-minimized", value); document.body.classList.toggle("hosted-panel-minimized", value);
            toggle.textContent = value ? "+" : "−"; toggle.setAttribute("aria-expanded", String(!value));
            toggle.setAttribute("aria-label", value ? "Show annotation controls" : "Minimize annotation controls"); toggle.title = toggle.getAttribute("aria-label");
            if (remember) try { sessionStorage.setItem("tator.hosted.panelMinimized", value ? "1" : "0"); } catch (_) { /* Per-tab convenience only. */ }
            window.dispatchEvent(new Event("resize"));
        };
        toggle.onclick = () => minimize(!content.hidden);
        let startMinimized = false;
        try { startMinimized = sessionStorage.getItem("tator.hosted.panelMinimized") === "1"; } catch (_) { /* Default to expanded. */ }
        const trancheInstructions = node("p"); trancheInstructions.id = "hostedTrancheInstructions"; trancheInstructions.style.whiteSpace = "pre-wrap"; content.append(trancheInstructions);
        const notes = node("div"); notes.id = "hostedReviewNotes"; content.append(notes);
        // Editing actions that otherwise need a keyboard (tablets) or are hard
        // to discover: undo, redo, delete the selected box, and the help sheet.
        const edits = node("div"); edits.className = "hosted-canvas-edit"; edits.setAttribute("role", "group"); edits.setAttribute("aria-label", "Edit this image");
        for (const [id, label, title, action] of [
            ["hostedUndo", "Undo", `Undo the last change on this image (${keyName("Z")})`, () => stepUndo(-1)],
            ["hostedRedo", "Redo", `Redo (${mac ? "⇧⌘Z" : "Ctrl+Y"})`, () => stepUndo(1)],
            ["hostedDeleteBox", "Delete box", "Delete the selected box (Delete, Backspace or X)", deleteSelection],
            ["hostedHelp", "Help & keys", "Mouse, touch and keyboard help (?)", () => toggleHelp(true)],
        ]) { const button = node("button", label); button.type = "button"; button.id = id; button.title = title; button.onclick = action; edits.append(button); }
        content.append(edits);
        const bar = node("div"); bar.className = "hosted-canvas-bar"; content.append(bar);
        addButton(bar, "hostedSave", "Save now", () => save({ manual: true }));
        addButton(bar, "hostedRefreshQueue", "Refresh work list", async () => { await queue.refresh(imageId); updateTranche((await queue.authorize(imageId)).tranche); status("Work list updated. Your current annotations are unchanged."); });
        addButton(bar, "hostedMoreImages", "Load more in this queue", async () => { await queue.more(); status(queue.next ? "More images added to the image list." : "All later images in this queue are loaded."); });
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
                await api(`${base}/override`, { ...context(), feedback: "Administrator recovered the image from the assignment." }); location.assign(`/annotate?${new URLSearchParams({image: imageId, queue: "unallocated"})}`);
            });
        }
        addButton(feedbackPanel, "hostedClassRequest", "Request a class", () => $("hostedClassDialog").showModal());
        const dialog = node("dialog"); dialog.id = "hostedClassDialog";
        const form = node("form"), name = node("input"), reason = node("textarea"); name.required = true; name.placeholder = "Class name"; name.setAttribute("aria-label", "Requested class name"); reason.required = true; reason.placeholder = "Why is this class needed?"; reason.setAttribute("aria-label", "Reason for class request");
        form.append(node("h2", "Request a dataset class"), name, reason);
        addButton(form, "hostedSendClass", "Send request", async () => { if (!form.reportValidity()) return; await api(`/api/hosted/datasets/${dataset.id}/class-requests`, { name: name.value, reason: reason.value, operation_id: uuid() }); dialog.close(); status("Class request sent. Existing classes remain available."); });
        addButton(form, "hostedCancelClass", "Cancel", () => dialog.close()); form.onsubmit = (e) => e.preventDefault(); dialog.append(form);
        content.append(feedbackPanel); panel.append(header, title, statusPanel, content);
        const narrow = node("div"); narrow.className = "hosted-narrow-notice"; narrow.setAttribute("role", "note");
        narrow.append(node("strong", "Annotation needs a larger screen"), node("p", "Open this image on a computer or a tablet at least 768 pixels wide. Your saved work is kept."));
        addButton(narrow, "hostedNarrowBack", "← Back to the workspace", () => $("hostedBack").click());
        document.body.append(panel, dialog, narrow, buildHelp());
        $("canvas")?.setAttribute("aria-label", "Image to annotate. Drag to draw a box in the selected class; click a box to select it.");
        installTouch();
        minimize(startMinimized, false);
        updateTranche(tranche);
        window.dispatchEvent(new Event("tator:hosted-canvas"));
        trackAdminNavHeight();
        update();
    }
    function buildHelp() {
        const help = node("dialog"); help.id = "hostedHelpDialog"; help.setAttribute("aria-labelledby", "hostedHelpTitle");
        const title = node("h2", "Annotating: mouse, touch and keys"); title.id = "hostedHelpTitle";
        const finish = boot.user.role === "annotator" ? "Submit & next" : "Approve & next";
        const basics = node("ul"); basics.className = "hosted-help-basics";
        for (const [what, how] of [
            ["Draw a box", "Choose a class under Classes, then drag on the image. On a touch screen, drag with one finger."],
            ["Select, move, resize", "Click or tap a box to select it. Drag it to move it, or drag a corner to resize it."],
            ["Change a box's class", "Select the box, then choose the class under Classes."],
            ["Delete", "Select the box, then press Delete, Backspace or X, or use Delete box. Q deletes the newest box."],
            ["Undo and redo", `${keyName("Z")} undoes the last change on this image; ${mac ? "⇧⌘Z" : "Ctrl+Y or Ctrl+Shift+Z"} redoes it. Undo and Redo are also in the panel.`],
            ["Polygons", "Press P or use Draw polygons, click each corner, and double-click to close the shape."],
            ["Zoom and pan", `Mouse wheel or pinch zooms; Shift + wheel, right-drag or two fingers pan. ${keyName("0")} fits the image again.`],
            ["Save", `Changes save automatically every few seconds; ${keyName("S")} or Save now saves at once.`],
            ["Other images", "→ or Space opens the next image and ← the previous one without handing anything over, or click a name in the image list. Tab moves between buttons and fields."],
            ["Finish an image", `Tick Annotations checked (or No objects to label), then ${finish} (${keyName("Enter")}).`],
        ]) { const item = node("li"); item.append(node("strong", `${what}: `), document.createTextNode(how)); basics.append(item); }
        help.append(title, basics);
        // The live list follows the keys set in this browser.
        const live = $("description");
        if (live) { const heading = node("h3", "All keyboard shortcuts"); help.append(heading, live); }
        const actions = node("div"); actions.className = "dialog-actions";
        const close = node("button", "Close"); close.type = "button"; close.autofocus = true; close.onclick = () => help.close(); actions.append(close); help.append(actions);
        help.addEventListener("click", event => { if (event.target === help) help.close(); });
        return help;
    }
    // Touch screens: one finger draws, moves and selects like the mouse (a
    // drag starts after a few pixels, a tap clicks, a double tap closes a
    // polygon); two fingers pinch to zoom and move to pan the image.
    function installTouch() {
        const canvas = $("canvas");
        if (!canvas || canvas.dataset.hostedTouch) return;
        canvas.dataset.hostedTouch = "true";
        let single = null, gesture = null, waitForLift = false, lastTap = null;
        const at = (touches) => ({ clientX: [...touches].reduce((sum, t) => sum + t.clientX, 0) / touches.length, clientY: [...touches].reduce((sum, t) => sum + t.clientY, 0) / touches.length });
        const spread = (touches) => Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY) || 1;
        const fire = (type, point, detail = 1) => canvas.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window,
            clientX: point.clientX, clientY: point.clientY, button: 0, buttons: type === "mouseup" ? 0 : 1, detail }));
        canvas.addEventListener("touchstart", event => {
            event.preventDefault();
            if (event.touches.length >= 2) {
                if (single?.down) fire("mouseup", single.last);
                single = null;
                const middle = at([...event.touches].slice(0, 2));
                gesture = { start: window.TatorHostedCanvas.beginGesture(middle.clientX, middle.clientY), spread: spread(event.touches) };
                return;
            }
            if (waitForLift || gesture) return;
            const touch = event.touches[0];
            single = { start: { clientX: touch.clientX, clientY: touch.clientY }, last: { clientX: touch.clientX, clientY: touch.clientY }, down: false };
            fire("mousemove", single.start, 0);
        }, { passive: false });
        canvas.addEventListener("touchmove", event => {
            event.preventDefault();
            if (gesture && event.touches.length >= 2) {
                const middle = at([...event.touches].slice(0, 2));
                window.TatorHostedCanvas.moveGesture(gesture.start, middle.clientX, middle.clientY, spread(event.touches) / gesture.spread);
                return;
            }
            if (!single) return;
            const touch = event.touches[0];
            single.last = { clientX: touch.clientX, clientY: touch.clientY };
            if (!single.down && Math.hypot(touch.clientX - single.start.clientX, touch.clientY - single.start.clientY) < 6) return;
            if (!single.down) { single.down = true; fire("mousedown", single.start); }
            fire("mousemove", single.last);
        }, { passive: false });
        const end = event => {
            event.preventDefault();
            if (gesture) { if (event.touches.length < 2) { gesture = null; waitForLift = event.touches.length > 0; } return; }
            if (!event.touches.length) waitForLift = false;
            if (!single) return;
            if (single.down) fire("mouseup", single.last);
            else {
                const now = Date.now(), point = single.start;
                const double = lastTap && now - lastTap.time < 400 && Math.hypot(point.clientX - lastTap.clientX, point.clientY - lastTap.clientY) < 24;
                fire("mousedown", point, double ? 2 : 1); fire("mouseup", point, double ? 2 : 1);
                lastTap = double ? null : { ...point, time: now };
            }
            single = null;
        };
        canvas.addEventListener("touchend", end, { passive: false });
        canvas.addEventListener("touchcancel", end, { passive: false });
    }
    function toggleHelp(open = !$("hostedHelpDialog")?.open) {
        const help = $("hostedHelpDialog"); if (!help) return;
        if (open && !help.open) help.showModal(); else if (!open && help.open) help.close();
    }
    // Keys the hosted canvas adds: save, undo/redo, hand over, fit, help.
    window.addEventListener("keydown", event => {
        if (!$("hostedSaveState") || event.defaultPrevented) return;
        const target = event.target instanceof Element ? event.target : null;
        if (target?.closest(".shortcut-settings-body")) return; // Recording a new binding.
        const typing =Boolean(target?.closest("textarea, select, [contenteditable=''], [contenteditable='true'], input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=range])"));
        const modifier = (event.ctrlKey || event.metaKey) && !event.altKey;
        const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
        const openDialog = document.querySelector("dialog[open]");
        // Escape belongs to an open dialog (the browser closes it); the
        // canvas's own Escape shortcut would otherwise swallow it.
        if (event.key === "Escape" && openDialog) { event.stopPropagation(); return; }
        let handled = true;
        if (modifier && key === "s") { if (ready) run(null, () => save({ manual: true })); }
        else if (openDialog && openDialog.id !== "hostedHelpDialog") handled = false;
        else if (event.key === "?" && !modifier && !typing) toggleHelp();
        else if (openDialog || typing || !modifier) handled = false;
        else if (key === "z") stepUndo(event.shiftKey ? 1 : -1);
        else if (key === "y" && !event.shiftKey) stepUndo(1);
        else if (key === "0") window.TatorHostedCanvas.fitView();
        else if (event.key === "Enter") {
            const finish = [$("hostedSubmit"), $("hostedApprove")].find(button => button && !button.hidden);
            if (finish && !finish.disabled) finish.click();
            else status(finish ? "Finish saving or reconnect editing before handing this image over." : "This image cannot be handed over from here.", true);
        }
        else handled = false;
        if (handled) { event.preventDefault(); event.stopPropagation(); }
    }, true);
    function trackAdminNavHeight() {
        // Lay the canvas out under the administrator navigation strip.
        const nav = $("hostedAdminNav"), root = document.documentElement;
        const sync = () => { root.style.setProperty("--hosted-nav-height", `${nav && !nav.hidden ? nav.getBoundingClientRect().height : 0}px`); window.dispatchEvent(new Event("resize")); };
        sync();
        if (nav && window.ResizeObserver) new ResizeObserver(sync).observe(nav);
    }
    function stateText(value = record) {
        const row = queue?.row(value.id);
        if (value.state === "confirmed") return "Confirmed";
        if (value.state === "submitted") return "Waiting for review";
        if (value.state === "returned") return value.assignee_id && value.assignee_id !== boot.user.user_id && row?.assignee_name ? `Returned for changes to ${row.assignee_name}` : "Returned for changes";
        if (value.state === "assigned") {
            if (value.assignee_id === boot.user.user_id || boot.user.role !== "admin") return "Assigned to you";
            return row?.assignee_name ? `Assigned to ${row.assignee_name}` : "Assigned to an annotator";
        }
        return value.tranche_id || row?.tranche_id ? "Not assigned" : "Incoming";
    }
    function backLabel() {
        const scope = queue?.scope || originScope;
        if (boot.user.role !== "admin") return scope?.tranche_id ? "← Back to tranche" : "← Workspace";
        const view = pageContext.view || (scope?.tranche_id ? "tranches" : {unallocated: "incoming", incoming: "incoming", confirmed: "confirmed", review: "submitted"}[scope?.queue]);
        if (view === "incoming") return "← Back to Incoming";
        if (view === "confirmed") return "← Back to Confirmed";
        if (view === "submitted" || view === "review") return "← Back to review";
        if (view === "tranches") return scope?.tranche_id ? "← Back to tranche" : "← Back to tranches";
        return "← Admin workspace";
    }
    async function start() {
        try {
            const sessionCheck = verifySession();
            await sessionCheck;
            const retained = sessionStorage.getItem(key);
            const recovery = retained ? JSON.parse(retained) : null;
            if (recovery?.scope) originScope = recovery.scope;
            if (recovery?.state === "committed_restricted") throw new Error("Your earlier change was saved, but its result is no longer available to this account. A later draft is retained in this tab. Ask an administrator to help recover it.");
            let recoveredReceipt = recovery?.pending ? await commandReceipt(recovery.pending) : null;
            if (recoveredReceipt && !recovery.pending.path.endsWith("/save")) {
                clearJournal(); location.assign(scopeBackUrl(recovery.scope || originScope)); return;
            }
            record = await api(`${base}?include_review=true`); dataset = await api(`/api/hosted/datasets/${record.dataset_id}`);
            const scoped = boot.scope ? {scope: boot.scope, tranche: boot.tranche} : await api(`${base}/scope`);
            scopeLeft = scopeLeft || Boolean(recovery?.scope && !sameScope(recovery.scope, scoped.scope));
            tranche = scoped.tranche;
            const neighbors = scopeLeft ? [record] : await openScope(scoped);
            buildUi();
            if (!retained && !scopeLeft) try { await acquire(false, false, record, sessionCheck); } catch (error) { report(error); }
            if (scopeLeft) {
                const notice = node("p", "This image left its previous tranche or queue. Back returns to that work list; your retained draft can be recovered only if you still have editing access and its saved version is unchanged.");
                notice.id = "hostedScopeNotice"; $("hostedSaveState").parentElement.append(notice);
                if (!retained) addButton($("hostedSaveState").parentElement, "hostedOpenCurrent", "Open current work queue", async () => {
                    const current = await api(`${base}/scope`);
                    location.assign(new window.HostedQueue({api, scope: current.scope}).url(imageId));
                });
            }
            await window.TatorHostedCanvas.open(record, dataset, !editable, neighbors);
            const start = Date.now();
            while (!window.TatorHostedCanvas.ready()) { if (Date.now() - start > 30000) throw new Error("Image loading did not finish. Reopen it to retry."); await new Promise((resolve) => setTimeout(resolve, 16)); }
            ready = true; lastSaved = JSON.stringify(draft()); resetUndo(); window.TatorHostedCanvas.freeze(!editable); update();
            if (retained) {
                recoveryRequired = true;
                freeze();
                status("A draft from the earlier connection is retained in this tab. Recover it before continuing.", true);
                addButton($("hostedSaveState").parentElement, "hostedResolve", "Recover my draft", async () => {
                    let receipt = recovery.pending ? await commandReceipt(recovery.pending) : null;
                    if (receipt && !recovery.pending.path.endsWith("/save")) { clearJournal(); location.assign(scopeBackUrl(recovery.scope || originScope)); return; }
                    let rejectedByLease = false, refusedHandoff = null;
                    if (recovery.pending && !receipt) {
                        // A missing receipt alone does not resolve a lost reply.
                        // Replay the original command before considering a new
                        // editing reservation or a different operation ID.
                        const original = recovery.pending;
                        try { receipt = {result: checkedResult(original, await api(original.path, original.body))}; }
                        catch (error) {
                            const certified = error.status === 422 && error.commit_outcome === "rejected" && error.operation_id === original.body.operation_id;
                            rejectedByLease = [409, 423].includes(error.status) && ["editor_lease_expired", "editor_lease_conflict", "ownership_changed", "annotation_version_changed"].includes(error.code);
                            // A refused submit/approve/return never committed once its
                            // receipt is still absent: restore the draft for correction
                            // instead of replaying the same refusal forever.
                            if (!certified && !rejectedByLease && !original.path.endsWith("/save") && definitiveRefusal(error)
                                && await commandReceipt(original).then(value => value === null, () => false)) refusedHandoff = error;
                            if (!certified && !rejectedByLease && !refusedHandoff) throw error;
                        }
                    }
                    if (receipt && recovery.pending && !recovery.pending.path.endsWith("/save")) { clearJournal(); location.assign(scopeBackUrl(recovery.scope || originScope)); return; }
                    const fresh = await api(base);
                    const anchor = recovery.pending?.body || recovery.unsent?.body || recovery.rejected?.context;
                    const expected = receipt?.result.current_version || anchor?.expected_version;
                    if (fresh.current_version !== expected) throw new Error("Someone saved a newer version. Your earlier draft is retained; it cannot replace their changes automatically. Ask the administrator to reconcile it.");
                    const restored = recovery.latest || recovery.pending?.body || recovery.unsent?.body;
                    if (!restored.annotations) throw new Error("No unsent annotations are attached to this operation. Check the workspace for its current review status.");
                    if (!confirm("Restore your retained draft on this unchanged server version? This also takes over your own previous editing tab.")) return;
                    record = fresh; await acquire(true, true);
                    // Recovery keeps an explicit queue (including global Review)
                    // while it remains valid. A derived scope is only a fallback
                    // for work that actually moved before or during recovery.
                    let replaceScope = scopeLeft || !queue;
                    if (!replaceScope) {
                        try { updateTranche((await queue.authorize(imageId)).tranche); }
                        catch (error) {
                            if (!["image_left_queue", "queue_scope_changed", "queue_scope_mismatch", "tranche_not_found"].includes(error.code)) throw error;
                            replaceScope = true;
                        }
                    }
                    if (replaceScope) {
                        const currentScope = await api(`${base}/scope`);
                        const neighbors = await openScope(currentScope); window.TatorHostedCanvas.reconcileQueue(neighbors);
                    }
                    history.replaceState(null, "", queue.url(imageId));
                    $("hostedScopeNotice")?.remove();
                    if (rejectedByLease && record.generation <= anchor.generation) throw new Error("The earlier editing reservation has not been replaced. Your original command and draft remain retained; reconnect before trying again.");
                    window.TatorHostedCanvas.restoreDraft(restored.annotations); resetUndo();
                    $("hostedComplete").checked = restored.complete;
                    $("hostedNegative").checked = restored.negative;
                    // Restoration is complete even if this draft still needs
                    // correction. Subsequent edits must use the live canvas,
                    // not the old recovery journal, as their source of truth.
                    recoveryRequired = false; $("hostedResolve").remove();
                    pending = null; await save({ manual: true });
                    if (refusedHandoff) status(`Your draft is restored. ${refusedHandoff.message}`, true);
                });
            } else if (editable) status("Connected · edits save automatically");
            else if (scopeLeft) status("This image left the previous queue. Open its current work queue to continue.", true);
            await refreshFeedback();
            scheduleMediaPrefetch();
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
            window.TatorHostedCanvas.restoreDraft(record.annotations); resetUndo();
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
        get session() { return accountChanged ? null : boot; },
        async leaveFor(destination) {
            const target = new URL(destination, location.href);
            if (target.origin !== location.origin || !["/", "/annotate", "/workbench"].includes(target.pathname)) throw new Error("Choose a hosted workspace destination.");
            const blocked = busy || handoff || recoveryRequired || accountChanged || !ready
                ? "Finish saving or reconnect editing before leaving this image."
                : window.TatorHostedCanvas.hasPendingTools() || [...activeTools.values()].some(item => !["sam_preload", "sam_state"].includes(item.operation))
                    ? "Wait for the current annotation tool, or cancel it before leaving." : null;
            if (blocked) { status(blocked, true); throw new Error(blocked); }
            busy = true; pauseMediaPrefetch(); update();
            try { await leave(false, target.href); }
            catch (error) { report(error); throw error; }
            finally { busy = false; update(); scheduleMediaPrefetch(); }
        },
        async prepareDatasetCaptionRun() {
            if (busy || handoff || recoveryRequired || accountChanged || !ready) throw new Error("Finish saving or reconnect editing before starting a dataset caption run.");
            if (boot.user.role !== "admin") throw new Error("An administrator starts dataset caption runs.");
            if (pending || saving || (!editable && JSON.stringify(draft()) !== lastSaved)) throw new Error("Save or recover the current image's unsent changes before starting a dataset caption run.");
            if (window.TatorHostedCanvas.hasPendingTools() || [...activeTools.values()].some(item => !["sam_preload", "sam_state"].includes(item.operation))) throw new Error("Wait for the annotation tool to finish, or cancel it before starting a dataset caption run.");
            busy = true; update();
            try { if (editable && !await save({manual: true})) throw new Error("Save the current image before starting a dataset caption run."); }
            finally { busy = false; update(); }
        },
        get editable() { return editable && ready && !handoff && !busy && !recoveryRequired; },
        editorId, get imageId() { return imageId; }, instanceId: boot.instance_id, get mediaUrl() { return `${base}/media`; }, save, navigate,
        mediaUrlFor: (id) => `/api/hosted/images/${encodeURIComponent(id)}/media`,
        mediaBlobFor,
        get mediaPrefetchState() { return mediaPrefetch?.stats || {entries: 0, bytes: 0, inFlight: 0}; },
        get datasetId() { return dataset?.id; }, api, step,
        get queueScope() { return queue?.scope; }, nextQueuedIds: count => queue?.peek(imageId, count) || [],
        async loadMoreQueue() { await queue?.more(); },
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
        objectAdded() {
            // Drawing on an image marked empty means it is no longer empty.
            const negative = $("hostedNegative");
            if (!negative?.checked || !ready || !editable) return;
            negative.checked = false;
            status("“No objects to label” was cleared because this image now has a box.");
        },
        notice(text, error = false) { status(text, error); },
        get samConfigured() { return Boolean(toolState?.capabilities?.sam_point?.configured); },
        get sam3PromptUnavailable() {
            const capability = toolState?.capabilities?.sam3_text;
            if (!capability || capability.configured) return null;
            return capability.reason === "tool_backend_not_supported"
                ? "This installed SAM3 model supports points and boxes. Text and similarity prompts need the native SAM3 weights on this server."
                : "SAM3 text and similarity prompts need a compatible SAM3 model selected in Tools & models.";
        },
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
    window.addEventListener("pageshow", event => {
        if (!event.persisted) return;
        // Back from analysis can restore a cached document whose editor lease
        // was released. A clean canvas must reload current versions/authority.
        clearMediaPrefetch();
        if (ready && !pending && !recoveryRequired && JSON.stringify(draft()) === lastSaved) location.reload();
        else { freeze(); status("Your draft is retained. Reconnect editing before continuing.", true); }
    });
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) pauseMediaPrefetch();
        else scheduleMediaPrefetch();
    });
    setInterval(() => {
        if (editable && ready && !busy && !handoff) {
            // Tool results arrive without a click or key: record them as their own undo step.
            noteEdit();
            if (saving && pending) { try { journal({ pending, latest: draft() }); } catch (error) { report(error); } }
            else save().catch(() => {});
        }
    }, 2000);
    setInterval(async () => {
        if (!record || !editable || busy || saving || handoff) return;
        const origin = {imageId, generation: record.generation};
        const current = () => imageId === origin.imageId && record?.generation === origin.generation && !busy && !handoff && !accountChanged;
        try {
            await verifySession(); if (!current()) return;
            const scoped = await queue.authorize(origin.imageId); if (!current()) return;
            if (scoped.tranche && scoped.tranche.revision !== tranche?.revision) await queue.refresh(origin.imageId, current);
            if (!current()) return;
            updateTranche(scoped.tranche);
            await api(`${base}/editor`, { editor_id: editorId, expected_version: record.current_version, generation: record.generation });
        } catch (error) { if (current()) report(error); }
    }, 30000);
    window.addEventListener("beforeunload", (event) => {
        // Undo may reach the saved baseline before the next autosave tick.
        // Reconcile that transition even when there is nothing left to send.
        try { retainDraft(); } catch (error) { status(error.message, true); }
        if (leavingToSignIn && !storageFailed) return; // The draft is retained for recovery after sign-in.
        if (storageFailed || pending || (ready && JSON.stringify(draft()) !== lastSaved)) {
            event.preventDefault(); event.returnValue = "";
        }
    });
    window.addEventListener("pagehide", () => {
        clearMediaPrefetch();
        if (!editable || !record || handoff || pending || saving) return;
        nativeFetch(`${base}/release`, { method: "POST", headers: { ...Object.fromEntries(headers()), "content-type": "application/json" }, body: JSON.stringify(context()), credentials: "same-origin", keepalive: true }).catch(() => {});
    });
})();
