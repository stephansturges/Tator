/* Resumable image batches. File bytes stay on disk; only retry identities persist. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const input = document.getElementById("uploadInput"), start = document.getElementById("import");
    input.accept += ",.txt";
    const resume = document.createElement("button"); resume.type = "button"; resume.id = "resumeImport"; resume.textContent = "Resume image import"; resume.hidden = true; start.after(resume);
    let resuming = false, running = false;
    const storageKey = (dataset) => `tator.hosted.import:${app.session?.instance_id}:${app.session?.user.user_id}:${dataset}`;
    function read(dataset) { try { return JSON.parse(sessionStorage.getItem(storageKey(dataset)) || "null"); } catch { return null; } }
    function persist(key, batch) { sessionStorage.setItem(key, JSON.stringify(batch)); }
    function refresh() { resume.hidden = running || app.session?.user.role !== "admin" || !read(app.datasetId); }
    function signature(file) { return JSON.stringify([file.name, file.size, file.lastModified]); }
    async function digest(file) { return [...new Uint8Array(await crypto.subtle.digest("SHA-256", await file.arrayBuffer()))].map((byte) => byte.toString(16).padStart(2, "0")).join(""); }
    start.onclick = () => { resuming = false; app.notice("Select images and, optionally, matching YOLO .txt labels plus labelmap.txt, classes.txt or labels.txt. Labels must use existing dataset classes."); input.click(); };
    resume.onclick = () => { resuming = true; app.notice("Select the same files again. Completed images will be skipped and interrupted uploads will use their original retry IDs."); input.click(); };
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
                const images = files.filter(file => /\.(png|jpe?g|webp|tiff?|bmp)$/i.test(file.name));
                const maps = files.filter(file => /^(labelmap|classes|labels)\.txt$/i.test(file.name));
                const sidecars = files.filter(file => /\.txt$/i.test(file.name) && !maps.includes(file));
                if (!images.length || files.length !== images.length + maps.length + sidecars.length) throw new Error("Select supported images and matching .txt labels only.");
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
                        const text = (await maps[0].text()).replace(/^\uFEFF/, "").trim();
                        names = text ? text.split(/\r?\n/).map(name => name.trim()) : [];
                        if (names.some(name => !name) || new Set(names.map(name => name.toLocaleLowerCase())).size !== names.length) throw new Error("The labelmap has empty or repeated class names.");
                        mapDigest = await digest(maps[0]);
                    } else names = description.classes.filter(item => !item.retired).sort((a, b) => a.position - b.position).map(item => item.name);
                    if (app.session !== authority || app.datasetId !== dataset) throw new Error("The account or dataset changed. Select these files again in the intended dataset.");
                    const order = names.map((name, index) => `${index}: ${name}`).join(", ");
                    if (!confirm(`Import ${sidecars.length} YOLO label file(s)? ${maps[0] ? "The selected labelmap" : "No labelmap was selected, so the current dataset order"} gives ${order || "no classes"}. Existing annotations are preserved for comparison if the images already exist.`)) { input.value = ""; return; }
                }
                batch = { class_names: names, expected_class_version: description?.class_version,
                    labelmap_signature: maps[0] ? signature(maps[0]) : null, labelmap_digest: mapDigest,
                    entries: images.map(file => ({ signature: signature(file), name: file.name, operation_id: crypto.randomUUID(), digest: null, state: "pending",
                        label_signature: labels.has(file.name.replace(/\.[^.]+$/, "")) ? signature(labels.get(file.name.replace(/\.[^.]+$/, ""))) : null, label_digest: null })) };
            } catch (error) { app.notice(error.message, true); input.value = ""; return; }
        }
        const work = [];
        for (const entry of batch.entries) {
            const file = candidates.get(entry.signature)?.shift();
            if (entry.state === "complete") continue;
            if (!file) { app.notice(`Reselect the remaining original files, including ${entry.name}.`, true); input.value = ""; return; }
            const label = entry.label_signature ? candidates.get(entry.label_signature)?.shift() : null;
            if (entry.label_signature && !label) { app.notice(`Reselect the original YOLO sidecar for ${entry.name}.`, true); input.value = ""; return; }
            work.push({ entry, file, label });
        }
        running = true; resume.hidden = true;
        await app.action(start, async () => {
            // Persist operation IDs before sending any bytes. A lost response
            // replays exactly the same server command after reselecting files.
            persist(key, batch);
            if (batch.labelmap_signature) {
                const originalMap = candidates.get(batch.labelmap_signature)?.[0];
                if (!originalMap || await digest(originalMap) !== batch.labelmap_digest) throw new Error("Reselect the unchanged original labelmap to resume this import.");
            }
            for (const { entry, file, label } of work) {
                if (file.size > 256 * 1024 * 1024) throw new Error(`${entry.name} exceeds the 256 MiB image limit.`);
                if (app.session !== authority || app.datasetId !== dataset) throw new Error("Import paused because the account or dataset changed. Resume it in its original dataset.");
                app.notice(`Adding ${batch.entries.filter((item) => item.state === "complete").length + 1} of ${batch.entries.length}: ${entry.name}`);
                const hash = await digest(file);
                if (entry.digest && entry.digest !== hash) throw new Error(`${entry.name} changed since the interrupted upload. Reselect its original bytes, or start a new import.`);
                entry.digest = hash; persist(key, batch);
                if (app.session !== authority || app.datasetId !== dataset) throw new Error("Import paused. Reopen its original dataset to resume.");
                const form = new FormData(); form.set("file", file); form.set("operation_id", entry.operation_id);
                if (label) {
                    const labelHash = await digest(label);
                    if (entry.label_digest && entry.label_digest !== labelHash) throw new Error(`The YOLO labels for ${entry.name} changed. Reselect their original bytes, or start a new import.`);
                    entry.label_digest = labelHash; persist(key, batch);
                    form.set("yolo_labels_json", JSON.stringify((await label.text()).replace(/^\uFEFF/, "")));
                    form.set("labelmap_names", JSON.stringify(batch.class_names)); form.set("expected_class_version", String(batch.expected_class_version));
                }
                if (app.session !== authority || app.datasetId !== dataset) throw new Error("Import paused because the account or dataset changed. Resume it in its original dataset.");
                const result = await app.api(`/datasets/${encodeURIComponent(dataset)}/incoming`, { method: "POST", body: form });
                if (!result || result.dataset_id !== dataset || result.media_hash !== hash || typeof result.id !== "string"
                        || typeof result.current_version !== "string" || !Number.isInteger(result.generation) || typeof result.duplicate !== "boolean"
                        || !(result.conflict_id === null || typeof result.conflict_id === "string")) throw new Error("The upload reply was incomplete. Resume this import to check its original result.");
                entry.state = "complete"; entry.duplicate = result.duplicate; entry.image_id = result.id; entry.conflict_id = result.conflict_id; persist(key, batch);
            }
            sessionStorage.removeItem(key);
            const existing = batch.entries.filter((entry) => entry.duplicate).length;
            const conflicts = batch.entries.filter(entry => entry.conflict_id);
            await app.refresh();
            app.notice(`${batch.entries.length - existing} new images added to Incoming. ${existing} existing copies preserved.${conflicts.length ? ` ${conflicts.length} imported label conflicts retained. Open these images and choose Imported labels to compare them: ${conflicts.map(entry => entry.name).join(", ")}.` : ""}`);
        });
        running = false; input.value = ""; refresh();
    };
    window.addEventListener("tator:hosted-session", refresh);
    document.getElementById("datasetChoice").addEventListener("change", refresh);
    setInterval(refresh, 3000);
    refresh();
})();
