/* Resumable image batches. File bytes stay on disk; only retry identities persist. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const input = document.getElementById("uploadInput"), start = document.getElementById("import");
    input.accept += ",.txt";
    const resume = document.createElement("button"); resume.type = "button"; resume.id = "resumeImport"; resume.textContent = "Resume image import"; resume.hidden = true; start.after(resume);
    // Stop finishes the image being sent; the rest stays resumable.
    const stop = document.createElement("button"); stop.type = "button"; stop.id = "stopImport"; stop.textContent = "Stop import"; stop.hidden = true; resume.after(stop);
    const frameLabel = document.createElement("label"), frame = document.createElement("select");
    frameLabel.textContent = "Label coordinates "; frame.id = "importLabelCoordinates";
    for (const [value, text] of [["", "Unspecified (unrotated)"], ["encoded", "Stored pixels"], ["exif_display", "Rotated viewer pixels"]]) {
        const option = document.createElement("option"); option.value = value; option.textContent = text; frame.append(option);
    }
    frameLabel.title = "Only needed when importing existing labels on photos with rotation metadata. Tator annotates the original stored pixel grid. Choose the grid your labels were drawn on; viewer coordinates will be converted without changing the original file.";
    frameLabel.append(frame); stop.after(frameLabel);
    // The last import's report stays under the Incoming actions in this tab until dismissed.
    const report = document.createElement("div"); report.id = "uploadReport"; report.className = "hosted-upload-report"; report.setAttribute("role", "status"); report.hidden = true;
    document.getElementById("viewActions").after(report);
    let resuming = false, running = false, stopping = false, destination = null;
    const storageKey = (dataset) => `tator.hosted.import:${app.session?.instance_id}:${app.session?.user.user_id}:${dataset}`;
    const reportKey = (dataset) => `tator.hosted.import-report:${app.session?.instance_id}:${app.session?.user.user_id}:${dataset}`;
    function read(dataset) { try { return JSON.parse(sessionStorage.getItem(storageKey(dataset)) || "null"); } catch { return null; } }
    function persist(key, batch) { sessionStorage.setItem(key, JSON.stringify(batch)); }
    function readReport(dataset) { try { return JSON.parse(sessionStorage.getItem(reportKey(dataset)) || "null"); } catch { return null; } }
    let drawn = "";
    function drawReport() {
        const saved = app.session?.user.role === "admin" && app.view === "incoming" ? readReport(app.datasetId) : null;
        // Redraw only on change, so the periodic refresh keeps keyboard focus.
        if (JSON.stringify(saved) === drawn) return;
        drawn = JSON.stringify(saved); report.replaceChildren(); report.hidden = !saved;
        if (!saved) return;
        report.dataset.error = String(Boolean(saved.error));
        const close = document.createElement("button"); close.type = "button"; close.textContent = "Dismiss"; close.setAttribute("aria-label", "Dismiss the import report");
        close.onclick = () => { try { sessionStorage.removeItem(reportKey(app.datasetId)); } catch (_) {} drawReport(); };
        const heading = document.createElement("strong"); heading.textContent = saved.title;
        report.append(heading);
        for (const line of saved.lines) { const item = document.createElement("p"); item.textContent = line; report.append(item); }
        report.append(close);
    }
    function refresh() {
        resume.hidden = running || app.session?.user.role !== "admin" || !read(app.datasetId);
        stop.hidden = !running; stop.disabled = stopping; frameLabel.hidden = start.hidden; frame.disabled = running;
        drawReport();
    }
    function signature(file) { return JSON.stringify([file.name, file.size, file.lastModified]); }
    async function digest(file) { return [...new Uint8Array(await crypto.subtle.digest("SHA-256", await file.arrayBuffer()))].map((byte) => byte.toString(16).padStart(2, "0")).join(""); }
    const plural = (n, word, many = `${word}s`) => `${n.toLocaleString()} ${n === 1 ? word : many}`;
    const IMAGE = /\.(png|jpe?g|webp|tiff?|bmp)$/i, LABELMAP = /^(labelmap|classes|labels)\.txt$/i;
    // The server refused this one file and saved nothing for it; the rest of the batch continues.
    const fileRejections = new Set(["invalid_image_import", "hosted_image_bytes_exceeded", "hosted_image_pixels_exceeded", "image_orientation_invalid", "image_encoded_dimensions_changed",
        "migration_label_invalid", "invalid_yolo_import", "label_coordinate_space_required", "label_coordinate_space_invalid", "invalid_bbox", "bbox_out_of_bounds", "invalid_class",
        "invalid_annotations", "invalid_polygon", "polygon_out_of_bounds", "duplicate_image_archived", "invalid_filename", "media_metadata_conflict", "request_too_large", "media_coordinate_information_required"]);
    function rejection(error) {
        const detail = error.detail;
        if (error.code === "invalid_image_import") return "not a readable image; the file may be damaged or in an unsupported format";
        if (error.code === "hosted_image_bytes_exceeded" || error.code === "request_too_large") return "larger than the 256 MiB limit";
        if (error.code === "hosted_image_pixels_exceeded") return "more than 64 million pixels";
        if (error.code === "image_orientation_invalid") return "its rotation metadata is invalid";
        if (error.code === "migration_label_invalid") return `label line ${detail?.line ?? "?"} ${detail?.reason === "unsupported YOLO row" ? "is not a YOLO box or polygon row" : "uses a class number that is not in the labelmap, or an invalid coordinate"}`;
        if (error.code === "label_coordinate_space_required") return "the photo has rotation metadata: choose Label coordinates, then add it again";
        if (error.code === "duplicate_image_archived") return "already in this dataset but archived; restore it from Export & recovery";
        return error.message.replace(/\.$/, "");
    }
    function launch(target = null) { destination = target?.tranche_id || null; resuming = false; app.notice("Select images and, optionally, matching YOLO .txt labels plus labelmap.txt, classes.txt or labels.txt. Labels must use existing dataset classes."); input.click(); }
    start.onclick = () => launch();
    window.TatorHostedImport = {start: launch};
    resume.onclick = () => { resuming = true; app.notice("Select the same files again. Completed images will be skipped and interrupted uploads will use their original retry IDs."); input.click(); };
    stop.onclick = () => { stopping = true; refresh(); app.notice("Stopping after the image being sent…"); };
    function summarise(batch, attachment, stopped) {
        const done = batch.entries.filter(entry => entry.state === "complete"), failed = batch.entries.filter(entry => entry.state === "failed");
        const remaining = batch.entries.length - done.length - failed.length, existing = done.filter(entry => entry.duplicate);
        const fresh = done.length - existing.length, lines = [];
        const names = (rows, limit = 6) => `${rows.slice(0, limit).join(", ")}${rows.length > limit ? `, and ${(rows.length - limit).toLocaleString()} more` : ""}`;
        const title = stopped
            ? `Import stopped: ${plural(done.length + failed.length, "file")} of ${batch.entries.length.toLocaleString()} handled, ${plural(remaining, "image")} not sent yet. Resume image import continues it; select the same files again.`
            : `${plural(fresh, "new image")} added${batch.destination_id ? "" : " to Incoming"}${failed.length ? `, ${failed.length.toLocaleString()} not added` : ""}.`;
        if (attachment) lines.push(`${plural(attachment.attached, "image")} attached to the tranche.${attachment.excluded.length ? ` Not attached: ${attachment.excluded.join("; ")}.` : ""}`);
        if (existing.length) lines.push(`${existing.length === 1 ? "1 image was" : `${existing.length.toLocaleString()} images were`} already in this dataset and kept as ${existing.length === 1 ? "it was" : "they were"}: ${names(existing.map(entry => entry.name))}.`);
        const sameName = done.filter(entry => entry.same_name);
        if (sameName.length) lines.push(`${sameName.length === 1 ? "1 new image has" : `${sameName.length.toLocaleString()} new images have`} the same file name as a different image already here; both are kept: ${names(sameName.map(entry => entry.name))}.`);
        const rows = key => done.reduce((sum, entry) => sum + (entry[key] || 0), 0), clipped = rows("clipped_rows"), dropped = rows("dropped_rows"), repeated = rows("duplicate_rows");
        if (clipped || dropped) lines.push(`Labels: ${[clipped ? `${plural(clipped, "box", "boxes")} clipped to the image edge` : "", repeated ? `${plural(repeated, "repeated row")} removed` : "", dropped - repeated ? `${plural(dropped - repeated, "row")} with no area inside the image dropped` : ""].filter(Boolean).join(", ")}.`);
        const conflicts = batch.entries.filter(entry => entry.conflict_id);
        if (conflicts.length) lines.push(`${conflicts.length === 1 ? "1 imported label conflict" : `${conflicts.length} imported label conflicts`} retained. Open these images and choose Imported labels to compare them: ${names(conflicts.map(entry => entry.name), 20)}.`);
        if (failed.length) lines.push(`Not added: ${names(failed.map(entry => `${entry.name} (${entry.error})`), 10)}. Fix ${failed.length === 1 ? "this file" : "these files"} and add ${failed.length === 1 ? "it" : "them"} again.`);
        if (batch.skipped?.length) lines.push(`Not image or label files, so not added: ${names(batch.skipped, 10)}.`);
        return {title, lines, error: Boolean(failed.length)};
    }
    function publish(dataset, result) {
        try { sessionStorage.setItem(reportKey(dataset), JSON.stringify(result)); } catch (_) {}
        refresh();
        // The headline replaces the progress line at the top; the details stay in the report.
        const shown = !report.hidden && app.datasetId === dataset;
        app.notice(shown ? `${result.title}${result.lines.length ? " Details are in the import report below." : ""}` : `${result.title} ${result.lines.join(" ")}`, result.error);
    }
    input.onchange = async (event) => {
        const files = [...event.target.files || []], dataset = app.datasetId, authority = app.session;
        if (!files.length || running || !dataset || authority?.user.role !== "admin") return;
        const key = storageKey(dataset);
        let batch = resuming ? read(dataset) : null;
        if (!resuming && read(dataset) && !confirm("Start a new import? Previously uploaded images stay preserved. You can reselect their files later without overwriting annotations.")) { input.value = ""; return; }
        if (resuming && !batch) { app.notice("There is no interrupted import for this dataset.", true); input.value = ""; return; }
        const candidates = new Map();
        for (const file of files) { const id = signature(file); if (!candidates.has(id)) candidates.set(id, []); candidates.get(id).push(file); }
        if (!batch) {
            try {
                const images = files.filter(file => IMAGE.test(file.name));
                if (destination && images.length > 500) throw new Error("Select at most 500 images when uploading into a tranche. No files were uploaded. You can add another batch afterwards.");
                const maps = files.filter(file => LABELMAP.test(file.name));
                const sidecars = files.filter(file => /\.txt$/i.test(file.name) && !maps.includes(file));
                const skipped = files.filter(file => !images.includes(file) && !maps.includes(file) && !sidecars.includes(file)).map(file => file.name);
                if (!images.length) {
                    if (files.some(file => /\.zip$/i.test(file.name))) throw new Error("ZIP archives are not unpacked in the browser. Unzip it, then select its images together with their .txt label files and labelmap. A whole dataset tree with captions or splits goes through the operator's migration tool instead.");
                    throw new Error("Select images (PNG, JPEG, WebP, TIFF or BMP), optionally with their .txt label files. No files were uploaded.");
                }
                if (maps.length > 1) throw new Error("Select one labelmap file, so the class order is unambiguous.");
                const stems = new Map();
                for (const file of images) { const stem = file.name.replace(/\.[^.]+$/, ""); if (!stems.has(stem)) stems.set(stem, []); stems.get(stem).push(file); }
                const labels = new Map();
                for (const file of sidecars) {
                    const stem = file.name.replace(/\.[^.]+$/, "");
                    if (stems.get(stem)?.length !== 1 || labels.has(stem)) throw new Error(`The label file ${file.name} does not match exactly one selected image.`);
                    if (file.size > 8 * 1024 * 1024) throw new Error(`${file.name} exceeds the 8 MiB label limit.`);
                    labels.set(stem, file);
                }
                if (maps.length && (!sidecars.length || stems.has(maps[0].name.replace(/\.[^.]+$/, "")))) throw new Error("The labelmap must accompany labelled images and cannot also be an image's sidecar.");
                let description = null, names = null, mapDigest = null;
                if (sidecars.length) {
                    description = await app.api(`/datasets/${encodeURIComponent(dataset)}`);
                    if (maps[0]) {
                        if (maps[0].size > 1024 * 1024) throw new Error("The labelmap exceeds 1 MiB.");
                        const text = (await maps[0].text()).replace(/^﻿/, "").trim();
                        names = text ? text.split(/\r?\n/).map(name => name.trim()) : [];
                        if (names.some(name => !name) || new Set(names.map(name => name.toLocaleLowerCase())).size !== names.length) throw new Error("The labelmap has empty or repeated class names.");
                        mapDigest = await digest(maps[0]);
                        const known = new Set(description.classes.filter(item => !item.retired).map(item => item.name.toLocaleLowerCase()));
                        const missing = names.filter(name => !known.has(name.toLocaleLowerCase()));
                        if (missing.length) throw new Error(`The labelmap names ${missing.length === 1 ? "a class" : "classes"} this dataset does not have: ${missing.join(", ")}. Add ${missing.length === 1 ? "it" : "them"} in Classes first, or select the labelmap that belongs to these labels. No files were uploaded.`);
                    } else names = description.classes.filter(item => !item.retired).sort((a, b) => a.position - b.position).map(item => item.name);
                    if (app.session !== authority || app.datasetId !== dataset) throw new Error("The account or dataset changed. Select these files again in the intended dataset.");
                    const order = names.map((name, index) => `${index}: ${name}`).join(", ");
                    if (!confirm(`Import ${sidecars.length} YOLO label file(s)? ${maps[0] ? "The selected labelmap" : "No labelmap was selected, so the current dataset order"} gives ${order || "no classes"}. Existing annotations are preserved for comparison if the images already exist.`)) { input.value = ""; return; }
                }
                batch = { destination_id: destination, class_names: names, expected_class_version: description?.class_version, label_coordinate_space: frame.value || null,
                    labelmap_signature: maps[0] ? signature(maps[0]) : null, labelmap_digest: mapDigest, skipped,
                    entries: images.map(file => ({ signature: signature(file), name: file.name, operation_id: crypto.randomUUID(), digest: null, state: "pending",
                        label_signature: labels.has(file.name.replace(/\.[^.]+$/, "")) ? signature(labels.get(file.name.replace(/\.[^.]+$/, ""))) : null, label_digest: null })) };
            } catch (error) { app.notice(error.message, true); input.value = ""; return; }
        }
        if (batch.destination_id && batch.entries.length > 500) {
            sessionStorage.removeItem(key); refresh(); input.value = "";
            app.notice("This older import contains more than 500 images. Completed uploads remain saved. Select batches of at most 500 for attachment; no further files were uploaded.", true); return;
        }
        const work = [];
        for (const entry of batch.entries) {
            const file = candidates.get(entry.signature)?.shift();
            if (entry.state === "complete" || entry.state === "failed") continue;
            if (!file) { app.notice(`Reselect the remaining original files, including ${entry.name}.`, true); input.value = ""; return; }
            const label = entry.label_signature ? candidates.get(entry.label_signature)?.shift() : null;
            if (entry.label_signature && !label) { app.notice(`Reselect the original YOLO sidecar for ${entry.name}.`, true); input.value = ""; return; }
            work.push({ entry, file, label });
        }
        running = true; stopping = false; refresh();
        try { sessionStorage.removeItem(reportKey(dataset)); } catch (_) {}
        let stopped = false;
        await app.action(start, async () => {
            // Persist operation IDs before sending any bytes. A lost response
            // replays exactly the same server command after reselecting files.
            persist(key, batch);
            if (batch.labelmap_signature && work.length) {
                const originalMap = candidates.get(batch.labelmap_signature)?.[0];
                if (!originalMap || await digest(originalMap) !== batch.labelmap_digest) throw new Error("Reselect the unchanged original labelmap to resume this import.");
            }
            const handled = () => batch.entries.filter((item) => item.state !== "pending").length;
            for (const { entry, file, label } of work) {
                if (stopping) { stopped = true; break; }
                if (app.session !== authority || app.datasetId !== dataset) throw new Error("Import paused because the account or dataset changed. Resume it in its original dataset.");
                app.notice(`Adding ${(handled() + 1).toLocaleString()} of ${batch.entries.length.toLocaleString()}: ${entry.name}`);
                if (file.size > 256 * 1024 * 1024) { entry.state = "failed"; entry.error = "larger than the 256 MiB limit"; persist(key, batch); continue; }
                const hash = await digest(file);
                if (entry.digest && entry.digest !== hash) throw new Error(`${entry.name} changed since the interrupted upload. Reselect its original bytes, or start a new import.`);
                entry.digest = hash; persist(key, batch);
                if (app.session !== authority || app.datasetId !== dataset) throw new Error("Import paused. Reopen its original dataset to resume.");
                const form = new FormData(); form.set("file", file); form.set("operation_id", entry.operation_id);
                if (label) {
                    const labelHash = await digest(label);
                    if (entry.label_digest && entry.label_digest !== labelHash) throw new Error(`The YOLO labels for ${entry.name} changed. Reselect their original bytes, or start a new import.`);
                    entry.label_digest = labelHash; persist(key, batch);
                    form.set("yolo_labels_json", JSON.stringify((await label.text()).replace(/^﻿/, "")));
                    form.set("labelmap_names", JSON.stringify(batch.class_names)); form.set("expected_class_version", String(batch.expected_class_version));
                    if (batch.label_coordinate_space) form.set("label_coordinate_space", batch.label_coordinate_space);
                }
                if (app.session !== authority || app.datasetId !== dataset) throw new Error("Import paused because the account or dataset changed. Resume it in its original dataset.");
                let result;
                try { result = await app.api(`/datasets/${encodeURIComponent(dataset)}/incoming`, { method: "POST", body: form }); }
                catch (error) {
                    if (error.stale || !fileRejections.has(error.code)) throw error;
                    entry.state = "failed"; entry.error = rejection(error); persist(key, batch); continue;
                }
                if (!result || result.dataset_id !== dataset || result.media_hash !== hash || typeof result.id !== "string"
                        || typeof result.current_version !== "string" || !Number.isInteger(result.generation) || typeof result.duplicate !== "boolean"
                        || !(result.conflict_id === null || typeof result.conflict_id === "string")) throw new Error("The upload reply was incomplete. Resume this import to check its original result.");
                entry.state = "complete"; entry.duplicate = result.duplicate; entry.image_id = result.id; entry.conflict_id = result.conflict_id;
                entry.same_name = Array.isArray(result.filename_collisions) && result.filename_collisions.length > 0;
                for (const name of ["clipped_rows", "dropped_rows", "duplicate_rows"]) if (Number.isInteger(result[name])) entry[name] = result[name];
                persist(key, batch);
                // Let the counts and an empty grid follow a long import instead of saying Incoming is empty.
                if (handled() % 25 === 0 && app.datasetId === dataset) {
                    app.refreshCounts();
                    if (app.view === "incoming" && !document.querySelector("#imageGrid .hosted-card")) app.loadImages().catch(() => {});
                }
            }
            if (stopped) { await app.refresh(); publish(dataset, summarise(batch, null, true)); return; }
            const attachment = batch.destination_id ? await window.TatorHostedTranches.attachImported(batch.destination_id, batch.entries.filter(entry => entry.state === "complete"), key) : null;
            sessionStorage.removeItem(key);
            await app.refresh();
            publish(dataset, summarise(batch, attachment, false));
        });
        running = false; stopping = false; input.value = ""; refresh();
    };
    window.addEventListener("tator:hosted-session", refresh);
    document.getElementById("datasetChoice").addEventListener("change", refresh);
    window.addEventListener("tator:hosted-dataset", refresh);
    window.addEventListener("tator:hosted-view", refresh);
    setInterval(refresh, 3000);
    refresh();
})();
