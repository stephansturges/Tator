(() => {
    "use strict";

    // -----------------------------------------
    // NEW CODE: Add a global dictionary for pending bboxes, and a UUID generator.
    // -----------------------------------------
    let pendingApiBboxes = {};

    const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]);
    const LABEL_EXTENSIONS = new Set(["txt"]);
    let directoryFallbackWarned = false;

    function generateUUID() {
        // Modern browsers: use crypto.randomUUID().
        // Fallback if unavailable.
        if (window.crypto && window.crypto.randomUUID) {
            return crypto.randomUUID();
        }
        // Fallback: timestamp + random
        return (
            Date.now().toString(36) +
            Math.random().toString(36).substring(2, 12)
        );
    }

    // 1) Define a palette of 100 colors spread around the hue wheel
    //    (First ~20 are roughly 0°, 18°, 36°, ..., 342°, then continuing).
    const colorPalette = [];
    for (let i = 0; i < 100; i++) {
        const baseHue = i * 20; 
        const randomOffset = Math.random() * 0.3;
        const hue = (baseHue + randomOffset) % 360;
        colorPalette.push(`hsla(${hue}, 100%, 45%, 1)`);
    }

    function getColorFromClass(className) {
        const index = classes[className] % 100; 
        return colorPalette[index];
    }

    function withAlpha(color, alpha) {
        return color.replace(/(\d?\.?\d+)\)$/, `${alpha})`);
    }

    let clipProgressFill = null;
    let clipProgressTimer = null;
    let clipProgressToken = 0;

    function ensureClipProgressElement() {
        if (!clipProgressFill) {
            clipProgressFill = document.getElementById("clipProgressFill");
        }
        return clipProgressFill;
    }

    function beginClipProgress() {
        const fill = ensureClipProgressElement();
        const token = ++clipProgressToken;
        if (!fill) {
            return token;
        }
        if (clipProgressTimer) {
            clearInterval(clipProgressTimer);
            clipProgressTimer = null;
        }
        fill.style.transition = "width 0.15s ease-out";
        fill.style.width = "10%";
        let progress = 10;
        clipProgressTimer = setInterval(() => {
            if (clipProgressToken !== token) {
                clearInterval(clipProgressTimer);
                clipProgressTimer = null;
                return;
            }
            progress = Math.min(progress + (Math.random() * 18 + 4), 85);
            fill.style.width = `${progress}%`;
        }, 200);
        return token;
    }

    function endClipProgress(token) {
        if (clipProgressToken !== token) {
            return;
        }
        const fill = ensureClipProgressElement();
        if (!fill) {
            clipProgressToken++;
            return;
        }
        if (clipProgressTimer) {
            clearInterval(clipProgressTimer);
            clipProgressTimer = null;
        }
        fill.style.transition = "width 0.12s ease-out";
        fill.style.width = "100%";
        const currentToken = token;
        setTimeout(() => {
            if (clipProgressToken !== currentToken) {
                return;
            }
            fill.style.transition = "width 0.2s ease-in";
            fill.style.width = "0%";
            clipProgressToken++;
        }, 160);
    }

    let autoMode = false;
    let samMode = false;
    let pointMode = false;
    let multiPointMode = false;
    let samAutoMode = false;
    let samPointAutoMode = false;
    let samMultiPointAutoMode = false;

    let multiPointPending = false;
    let multiPointPendingToken = null;
    let multiPointPoints = [];
    let multiPointPendingBboxInfo = null;
    const multiPointQueue = [];
    let multiPointWaitingForPreload = false;

    let samVariant = "sam1";
    let autoModeCheckbox = null;
    let samModeCheckbox = null;
    let pointModeCheckbox = null;
    let multiPointModeCheckbox = null;
    let samVariantSelect = null;
    let samPreloadCheckbox = null;
    let samPreloadEnabled = false;
    let samPreloadToken = 0;
    let samPreloadAbortController = null;
    let samPreloadLastKey = null;
    let samPreloadCurrentImageName = null;
    let samStatusProgressEl = null;
    const samTokenCache = new Map();
    let samPreloadTimer = null;
    const SAM_PRELOAD_DEBOUNCE_MS = 250;
    const SAM_PRELOAD_IMAGE_SWITCH_DELAY_MS = 320;
    let samPreloadGeneration = 0;
    let samPreloadCurrentVariant = null;
    const SAM_PRELOAD_WAIT_TIMEOUT_MS = 8000;
    const samPreloadWatchers = new Map();

    let imagesSelectButton = null;
    let classesSelectButton = null;
    let bboxesSelectButton = null;

    let samStatusEl = null;
    let samStatusTimer = null;
    let samStatusMessageToken = 0;
    let samJobSequence = 0;
    let samCancelVersion = 0;
    const samActiveJobs = new Map();

    const multiPointColors = {
        positive: { stroke: "#2ecc71", fill: "rgba(46, 204, 113, 0.35)" },
        negative: { stroke: "#e74c3c", fill: "rgba(231, 76, 60, 0.35)" },
    };

    const API_ROOT = "http://localhost:8000";
    const TAB_LABELING = "labeling";
    const TAB_TRAINING = "training";
    const TAB_ACTIVE = "active";

    let activeTab = TAB_LABELING;
    let trainingUiInitialized = false;
    let activeUiInitialized = false;
    let loadedClassList = [];

    const tabElements = {
        labelingButton: null,
        trainingButton: null,
        activeButton: null,
        labelingPanel: null,
        trainingPanel: null,
        activePanel: null,
    };

    const trainingElements = {
        clipBackboneSelect: null,
        solverSelect: null,
        imagesInput: null,
        imagesBtn: null,
        imagesSummary: null,
        labelsInput: null,
        labelsBtn: null,
        labelsSummary: null,
        labelmapInput: null,
        labelmapSummary: null,
        outputDirBtn: null,
        outputDirSummary: null,
        modelFilenameInput: null,
        labelmapFilenameInput: null,
        testSizeInput: null,
        randomSeedInput: null,
        batchSizeInput: null,
        maxIterInput: null,
        minPerClassInput: null,
        regCInput: null,
        classWeightSelect: null,
        deviceOverrideInput: null,
        startButton: null,
        cancelButton: null,
        progressFill: null,
        statusText: null,
        message: null,
        summary: null,
        log: null,
        historyContainer: null,
        reuseEmbeddingsCheckbox: null,
        hardMiningCheckbox: null,
    };

    const activeElements = {
        message: null,
        info: null,
        clipSelect: null,
        classifierPath: null,
        classifierUpload: null,
        classifierBrowse: null,
        labelmapPath: null,
        labelmapUpload: null,
        labelmapBrowse: null,
        activateLatestButton: null,
        applyButton: null,
        refreshButton: null,
    };

    const trainingState = {
        activeJobId: null,
        pollHandle: null,
        jobs: new Map(),
        latestArtifacts: null,
        outputDirPath: ".",
        imagesFolderName: null,
        labelsFolderName: null,
        imageEntries: [],
        labelEntries: [],
        imageTotalCount: 0,
        labelTotalCount: 0,
        nativeImagesPath: null,
        nativeLabelsPath: null,
    };

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function formatTimestamp(seconds) {
        if (typeof seconds !== "number" || Number.isNaN(seconds)) {
            return "";
        }
        const date = new Date(seconds * 1000);
        if (Number.isNaN(date.getTime())) {
            return "";
        }
        return date.toLocaleTimeString([], { hour12: false });
    }

    function formatNumber(value, digits = 4) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return "—";
        }
        return Number(value).toFixed(digits);
    }

    function renderConvergenceTable(trace) {
        if (!Array.isArray(trace) || !trace.length) {
            return "";
        }
        const slice = trace.slice(-200);
        const header = `
            <tr>
                <th>Iter</th>
                <th>Train Loss</th>
                <th>Train Acc</th>
                <th>Val Loss</th>
                <th>Val Acc</th>
                <th>|Δw|</th>
            </tr>`;
        const rows = slice.map((entry) => {
            const iterRaw = entry.iteration ?? entry.iter ?? "";
            const iter = escapeHtml(String(iterRaw));
            const trainLoss = escapeHtml(formatNumber(entry.train_loss));
            const trainAcc = escapeHtml(formatNumber(entry.train_accuracy, 3));
            const valLoss = escapeHtml(formatNumber(entry.val_loss));
            const valAcc = escapeHtml(formatNumber(entry.val_accuracy, 3));
            const coefDelta = escapeHtml(formatNumber(entry.coef_delta));
            return `<tr>
                <td>${iter}</td>
                <td>${trainLoss}</td>
                <td>${trainAcc}</td>
                <td>${valLoss}</td>
                <td>${valAcc}</td>
                <td>${coefDelta}</td>
            </tr>`;
        }).join("");
        const note = slice.length < trace.length ? `<div class="training-convergence-note">Showing last ${slice.length} of ${trace.length} iterations.</div>` : "";
        return `<div class="training-convergence"><h4>Solver Progress</h4>${note}<table>${header}${rows}</table></div>`;
    }

    function renderPerClassMetrics(metrics) {
        if (!Array.isArray(metrics) || !metrics.length) {
            return "";
        }
        const rows = metrics.map((entry) => {
            const label = escapeHtml(String(entry.label ?? "class"));
            const precision = Number.isFinite(entry.precision) ? entry.precision * 100 : null;
            const recall = Number.isFinite(entry.recall) ? entry.recall * 100 : null;
            const f1 = Number.isFinite(entry.f1) ? entry.f1 * 100 : null;
            const support = Number.isFinite(entry.support) ? parseInt(entry.support, 10) : null;

            const precisionBar = precision !== null
                ? `<div class="metric-bar-track"><div class="metric-bar-fill precision" style="width:${Math.max(0, Math.min(100, precision)).toFixed(1)}%"></div></div><span class="metric-value">${precision.toFixed(1)}%</span>`
                : `<span class="metric-value">—</span>`;
            const recallBar = recall !== null
                ? `<div class="metric-bar-track"><div class="metric-bar-fill recall" style="width:${Math.max(0, Math.min(100, recall)).toFixed(1)}%"></div></div><span class="metric-value">${recall.toFixed(1)}%</span>`
                : `<span class="metric-value">—</span>`;
            const f1Bar = f1 !== null
                ? `<div class="metric-bar-track"><div class="metric-bar-fill f1" style="width:${Math.max(0, Math.min(100, f1)).toFixed(1)}%"></div></div><span class="metric-value">${f1.toFixed(1)}%</span>`
                : `<span class="metric-value">—</span>`;

            return `<div class="metric-row">
                <div class="metric-label">${label}</div>
                <div class="metric-measure">
                    <span class="metric-tag">P</span>
                    ${precisionBar}
                </div>
                <div class="metric-measure">
                    <span class="metric-tag">R</span>
                    ${recallBar}
                </div>
                <div class="metric-measure">
                    <span class="metric-tag">F1</span>
                    ${f1Bar}
                </div>
                <div class="metric-support">n=${support !== null ? escapeHtml(String(support)) : "—"}</div>
            </div>`;
        }).join("");

        const chart = renderPerClassMetricChart(metrics);
        const glossary = renderMetricGlossary();

        return `<div class="per-class-metrics"><h4>Per-class Metrics</h4><div class="metric-rows">${rows}</div>${chart}${glossary}</div>`;
    }

    function renderPerClassMetricChart(metrics) {
        if (!Array.isArray(metrics) || !metrics.length) {
            return "";
        }

        const columns = metrics.map((entry) => {
            const label = escapeHtml(String(entry.label ?? "class"));
            const precision = Number.isFinite(entry.precision) ? Math.max(0, Math.min(100, entry.precision * 100)) : null;
            const recall = Number.isFinite(entry.recall) ? Math.max(0, Math.min(100, entry.recall * 100)) : null;
            const f1 = Number.isFinite(entry.f1) ? Math.max(0, Math.min(100, entry.f1 * 100)) : null;

            const precisionBar = precision === null
                ? '<div class="metric-chart-bar precision empty"></div>'
                : `<div class="metric-chart-bar precision" style="height:${precision.toFixed(1)}%" title="Precision: ${precision.toFixed(1)}%"></div>`;
            const recallBar = recall === null
                ? '<div class="metric-chart-bar recall empty"></div>'
                : `<div class="metric-chart-bar recall" style="height:${recall.toFixed(1)}%" title="Recall: ${recall.toFixed(1)}%"></div>`;
            const f1Bar = f1 === null
                ? '<div class="metric-chart-bar f1 empty"></div>'
                : `<div class="metric-chart-bar f1" style="height:${f1.toFixed(1)}%" title="F1: ${f1.toFixed(1)}%"></div>`;

            return `<div class="metric-chart-column">
                <div class="metric-chart-bars">
                    ${precisionBar}
                    ${recallBar}
                    ${f1Bar}
                </div>
                <div class="metric-chart-class" title="${label}">${label}</div>
            </div>`;
        }).join("");

        return `<div class="metric-chart">
            <h4>Per-class Score Chart</h4>
            <div class="metric-chart-grid">${columns}</div>
            <div class="metric-chart-legend">
                <div class="metric-chart-legend-item"><span class="legend-swatch precision"></span>Precision</div>
                <div class="metric-chart-legend-item"><span class="legend-swatch recall"></span>Recall</div>
                <div class="metric-chart-legend-item"><span class="legend-swatch f1"></span>F1</div>
            </div>
        </div>`;
    }

    function renderMetricGlossary() {
        return `<div class="metric-glossary">
            <strong>Precision</strong> measures how many of the model's positive predictions were correct. 
            <strong>Recall</strong> captures how many true examples the model managed to find. 
            <strong>F1</strong> balances precision and recall in a single score.
        </div>`;
    }

    function setTrainingMessage(text, variant) {
        const el = trainingElements.message;
        if (!el) {
            return;
        }
        el.textContent = text || "";
        el.classList.remove("error", "success", "warn");
        if (variant) {
            el.classList.add(variant);
        }
    }

    function setActiveMessage(text, variant) {
        const el = activeElements.message;
        if (!el) {
            return;
        }
        el.textContent = text || "";
        el.classList.remove("error", "success", "warn");
        if (variant) {
            el.classList.add(variant);
        }
    }

    function getRootFolderName(fileOrPath) {
        if (!fileOrPath) {
            return "";
        }
        const rel = typeof fileOrPath === "string"
            ? fileOrPath
            : fileOrPath.webkitRelativePath || fileOrPath.relativePath || fileOrPath.name;
        if (rel) {
            const parts = rel.split(/[\\/]/).filter(Boolean);
            if (parts.length) {
                return parts[0];
            }
        }
        return typeof fileOrPath === "string" ? fileOrPath : fileOrPath.name;
    }

    function updateFileSummary(inputEl, summaryEl, options = {}) {
        if (!summaryEl) {
            return;
        }
        const entries = Array.isArray(options.entries) ? options.entries : null;
        const files = entries
            ? entries.map((entry) => entry.file || entry)
            : inputEl && inputEl.files
                ? Array.from(inputEl.files)
                : [];
        if (!files.length) {
            summaryEl.textContent = options.emptyText || "No files selected";
            if (summaryEl.textContent) {
                summaryEl.title = summaryEl.textContent;
            } else {
                summaryEl.removeAttribute("title");
            }
            return;
        }
        if (options.mode === "path") {
            const pathText = options.path || "";
            summaryEl.textContent = pathText || (options.emptyText || "No path selected");
            if (summaryEl.textContent) {
                summaryEl.title = summaryEl.textContent;
            } else {
                summaryEl.removeAttribute("title");
            }
            return;
        }
        if (options.mode === "folder") {
            const allowedExts = options.allowedExts;
            const totalCount = typeof options.totalCount === "number" && options.totalCount >= files.length
                ? options.totalCount
                : files.length;
            let validCount;
            if (entries) {
                validCount = entries.length;
            } else if (allowedExts) {
                validCount = files.filter((file) => allowedExts.has((file.name.split(".").pop() || "").toLowerCase())).length;
            } else {
                validCount = files.length;
            }
            const folderName = options.folderName
                || (entries && entries.length
                    ? getRootFolderName(entries[0].relativePath || entries[0].file?.name)
                    : getRootFolderName(files[0]));
            const descriptor = validCount === totalCount
                ? `${validCount} files`
                : `${validCount} of ${totalCount} files`;
            summaryEl.textContent = `${folderName} (${descriptor})`;
            summaryEl.title = summaryEl.textContent;
            return;
        }
        if (files.length === 1) {
            summaryEl.textContent = files[0].name;
            summaryEl.title = summaryEl.textContent;
            return;
        }
        summaryEl.textContent = `${files.length} files selected`;
        summaryEl.title = summaryEl.textContent;
    }

    function stopTrainingPoll() {
        if (trainingState.pollHandle !== null) {
            clearTimeout(trainingState.pollHandle);
            trainingState.pollHandle = null;
        }
    }

    const HAS_DIRECTORY_PICKER = typeof window !== "undefined" && typeof window.showDirectoryPicker === "function";

    function getFileExtension(name) {
        const lower = (name || "").toLowerCase();
        const parts = lower.split(".");
        if (parts.length <= 1) {
            return "";
        }
        return parts.pop() || "";
    }

    async function collectDirectoryEntries(handle, allowedExts) {
        const collected = [];
        let totalCount = 0;

        async function walk(dirHandle, prefix) {
            for await (const entry of dirHandle.values()) {
                if (entry.kind === "file") {
                    totalCount += 1;
                    const file = await entry.getFile();
                    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
                    if (!allowedExts || allowedExts.has(getFileExtension(entry.name))) {
                        collected.push({ file, relativePath });
                    }
                } else if (entry.kind === "directory") {
                    const nextPrefix = prefix ? `${prefix}/${entry.name}` : entry.name;
                    await walk(entry, nextPrefix);
                }
            }
        }

        await walk(handle, "");
        return { entries: collected, totalCount };
    }

    function getStoredEntries(kind) {
        if (kind === "images") {
            if (Array.isArray(trainingState.imageEntries) && trainingState.imageEntries.length) {
                return trainingState.imageEntries.slice();
            }
        } else if (Array.isArray(trainingState.labelEntries) && trainingState.labelEntries.length) {
            return trainingState.labelEntries.slice();
        }
        const input = kind === "images" ? trainingElements.imagesInput : trainingElements.labelsInput;
        const allowed = kind === "images" ? IMAGE_EXTENSIONS : LABEL_EXTENSIONS;
        if (!input || !input.files) {
            return [];
        }
        return Array.from(input.files)
            .filter((file) => allowed.has(getFileExtension(file.name)))
            .map((file) => ({
                file,
                relativePath: file.webkitRelativePath || file.relativePath || file.name,
            }));
    }

    async function handleNativeFolderFallback(kind) {
        const previousPath = kind === "images" ? trainingState.nativeImagesPath : trainingState.nativeLabelsPath;
        const params = new URLSearchParams();
        if (previousPath) {
            params.set('initial', previousPath);
        }
        const endpoint = `${API_ROOT}/fs/select_directory${params.toString() ? `?${params.toString()}` : ''}`;
        try {
            const resp = await fetch(endpoint);
            if (!resp.ok) {
                let detail;
                try {
                    detail = (await resp.json()).detail;
                } catch (err) {
                    detail = await resp.text();
                }
                if (detail === "tkinter_unavailable") {
                    await promptForFolderPath(kind);
                    return;
                }
                throw new Error(detail || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const selected = data && data.path ? data.path : null;
            if (!selected) {
                setTrainingMessage("Directory selection cancelled.", null);
                return;
            }
            applyNativeFolderSelection(kind, selected);
        } catch (error) {
            console.warn('Native directory picker failed', error);
            await promptForFolderPath(kind, error);
        }
    }

    async function promptForFolderPath(kind, reason) {
        if (reason && reason !== 'prompt') {
            setTrainingMessage(`Directory picker unavailable (${reason.message || reason}).`, 'warn');
        }
        const previousPath = kind === "images" ? trainingState.nativeImagesPath : trainingState.nativeLabelsPath;
        const promptLabel = kind === "images" ? "Enter images folder path" : "Enter labels folder path";
        const entered = window.prompt(promptLabel, previousPath || "");
        if (entered === null) {
            setTrainingMessage("Directory selection cancelled.", null);
            return;
        }
        const trimmed = entered.trim();
        if (!trimmed) {
            setTrainingMessage("Folder path cannot be empty.", "error");
            return;
        }
        applyNativeFolderSelection(kind, trimmed);
    }

    function applyNativeFolderSelection(kind, absolutePath) {
        const folderName = getRootFolderName(absolutePath);
        let successText;
        if (kind === "images") {
            trainingState.nativeImagesPath = absolutePath;
            trainingState.imageEntries = [];
            trainingState.imageTotalCount = 0;
            trainingState.imagesFolderName = folderName;
            updateFileSummary(null, trainingElements.imagesSummary, { mode: "path", path: absolutePath, emptyText: "No folder selected" });
            if (trainingElements.imagesSummary) {
                trainingElements.imagesSummary.textContent = absolutePath;
                trainingElements.imagesSummary.title = absolutePath;
            }
            successText = `Using server-side images folder: ${absolutePath}`;
        } else {
            trainingState.nativeLabelsPath = absolutePath;
            trainingState.labelEntries = [];
            trainingState.labelTotalCount = 0;
            trainingState.labelsFolderName = folderName;
            updateFileSummary(null, trainingElements.labelsSummary, { mode: "path", path: absolutePath, emptyText: "No folder selected" });
            if (trainingElements.labelsSummary) {
                trainingElements.labelsSummary.textContent = absolutePath;
                trainingElements.labelsSummary.title = absolutePath;
            }
            successText = `Using server-side labels folder: ${absolutePath}`;
        }
        if (!HAS_DIRECTORY_PICKER && !directoryFallbackWarned) {
            directoryFallbackWarned = true;
            setTrainingMessage("Using the classic folder dialog (expected when opening ybat.html directly).", null);
        }
        if (successText) {
            setTrainingMessage(successText, "success");
        }
    }

    function summariseEntries(kind, entries, totalCount, folderName) {
        if (kind === "images") {
            trainingState.imageEntries = entries;
            trainingState.imageTotalCount = totalCount;
            trainingState.imagesFolderName = folderName;
            trainingState.nativeImagesPath = null;
            updateFileSummary(null, trainingElements.imagesSummary, {
                emptyText: "No folder selected",
                mode: "folder",
                allowedExts: IMAGE_EXTENSIONS,
                entries,
                totalCount,
                folderName,
            });
        } else {
            trainingState.labelEntries = entries;
            trainingState.labelTotalCount = totalCount;
            trainingState.labelsFolderName = folderName;
            trainingState.nativeLabelsPath = null;
            updateFileSummary(null, trainingElements.labelsSummary, {
                emptyText: "No folder selected",
                mode: "folder",
                allowedExts: LABEL_EXTENSIONS,
                entries,
                totalCount,
                folderName,
            });
        }
    }

    async function chooseTrainingFolder(kind) {
        if (!HAS_DIRECTORY_PICKER) {
            handleNativeFolderFallback(kind);
            return;
        }
        const allowed = kind === "images" ? IMAGE_EXTENSIONS : LABEL_EXTENSIONS;
        try {
            const handle = await window.showDirectoryPicker();
            if (!handle) {
                return;
            }
            const { entries, totalCount } = await collectDirectoryEntries(handle, allowed);
            if (!entries.length) {
                throw new Error("Selected folder does not contain supported files.");
            }
            summariseEntries(kind, entries, totalCount, handle.name);
            setTrainingMessage(`Loaded ${entries.length} ${kind === "images" ? "image" : "label"} files from ${handle.name}.`, "success");
        } catch (error) {
            if (error && (error.name === "AbortError" || error.name === "NotAllowedError")) {
                setTrainingMessage("Directory selection cancelled.", null);
                return;
            }
            console.error("Directory picker failed", error);
            setTrainingMessage(error.message || String(error), "error");
        }
    }

    function handleImagesInputChange() {
        if (!trainingElements.imagesInput) {
            return;
        }
        const rawFiles = Array.from(trainingElements.imagesInput.files || []);
        if (!rawFiles.length) {
            summariseEntries("images", [], 0, null);
            trainingState.nativeImagesPath = null;
            return;
        }
        const entries = rawFiles
            .filter((file) => IMAGE_EXTENSIONS.has(getFileExtension(file.name)))
            .map((file) => ({
                file,
                relativePath: file.webkitRelativePath || file.relativePath || file.name,
            }));
        summariseEntries("images", entries, rawFiles.length, getRootFolderName(rawFiles[0]));
        trainingState.nativeImagesPath = null;
        if (trainingElements.reuseEmbeddingsCheckbox) {
            trainingElements.reuseEmbeddingsCheckbox.checked = false;
        }
        if (!entries.length) {
            setTrainingMessage("No supported image files found in folder.", "error");
        }
    }

    function handleLabelsInputChange() {
        if (!trainingElements.labelsInput) {
            return;
        }
        const rawFiles = Array.from(trainingElements.labelsInput.files || []);
        if (!rawFiles.length) {
            summariseEntries("labels", [], 0, null);
            trainingState.nativeLabelsPath = null;
            return;
        }
        const entries = rawFiles
            .filter((file) => LABEL_EXTENSIONS.has(getFileExtension(file.name)))
            .map((file) => ({
                file,
                relativePath: file.webkitRelativePath || file.relativePath || file.name,
            }));
        summariseEntries("labels", entries, rawFiles.length, getRootFolderName(rawFiles[0]));
        trainingState.nativeLabelsPath = null;
        if (trainingElements.reuseEmbeddingsCheckbox) {
            trainingElements.reuseEmbeddingsCheckbox.checked = false;
        }
        if (!entries.length) {
            setTrainingMessage("No YOLO label files found in folder.", "error");
        }
    }

    function startTrainingPoll(jobId, immediate = false) {
        stopTrainingPoll();
        trainingState.activeJobId = jobId;
        const poll = () => {
            pollTrainingJob(jobId).catch((err) => {
                console.error("Failed to poll training job", err);
                setTrainingMessage(`Failed to poll training job: ${err.message || err}`, "error");
            });
        };
        if (immediate) {
            poll();
        } else {
            trainingState.pollHandle = setTimeout(poll, 1500);
        }
    }

    function fillSelectOptions(selectEl, options, preferred) {
        if (!selectEl) {
            return;
        }
        const previous = selectEl.value;
        selectEl.innerHTML = "";
        options.forEach((name) => {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            selectEl.appendChild(option);
        });
        const desired = preferred || previous || (options.length ? options[0] : "");
        if (desired && options.includes(desired)) {
            selectEl.value = desired;
        }
    }

    async function populateClipBackbones() {
        try {
            const resp = await fetch(`${API_ROOT}/clip/backbones`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const list = Array.isArray(data.available) ? data.available : [];
            const active = data.active || (list.length ? list[0] : null);
            fillSelectOptions(trainingElements.clipBackboneSelect, list, active);
            fillSelectOptions(activeElements.clipSelect, list, active);
        } catch (error) {
            console.warn("Failed to fetch clip backbones", error);
            setTrainingMessage(`Unable to load CLIP backbones: ${error.message || error}`, "error");
            setActiveMessage(`Unable to load CLIP backbones: ${error.message || error}`, "error");
        }
    }

    async function refreshActiveModelPanel() {
        try {
            const resp = await fetch(`${API_ROOT}/clip/active_model`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const clipModelName = data.clip_model || "(not loaded)";
            const classifierPath = data.classifier_path || "(none)";
            const labelmapPath = data.labelmap_path || "(none)";
            if (activeElements.info) {
                activeElements.info.innerHTML = `CLIP: <strong>${escapeHtml(clipModelName)}</strong><br/>Classifier: <strong>${escapeHtml(classifierPath)}</strong><br/>Labelmap: <strong>${escapeHtml(labelmapPath)}</strong>`;
            }
            if (activeElements.classifierPath) {
                activeElements.classifierPath.value = data.classifier_path || "";
            }
            if (activeElements.labelmapPath) {
                activeElements.labelmapPath.value = data.labelmap_path || "";
            }
            if (activeElements.clipSelect && data.clip_model) {
                activeElements.clipSelect.value = data.clip_model;
            }
            if (trainingElements.clipBackboneSelect && data.clip_model) {
                trainingElements.clipBackboneSelect.value = data.clip_model;
            }
            setActiveMessage(data.clip_ready ? "" : "CLIP classifier is not ready. Load a model to enable auto-labeling.", data.clip_ready ? null : "error");
        } catch (error) {
            console.warn("Failed to refresh active model", error);
            setActiveMessage(`Unable to read active model: ${error.message || error}`, "error");
        }
    }
    function renderTrainingHistoryItem(container, job) {
        const item = document.createElement("div");
        item.className = "training-history-item";
        const left = document.createElement("div");
        left.textContent = `${job.job_id.slice(0, 8)}… — ${job.status}`;
        const right = document.createElement("div");
        const viewBtn = document.createElement("button");
        viewBtn.type = "button";
        viewBtn.textContent = "View";
        viewBtn.addEventListener("click", () => {
            loadTrainingJob(job.job_id, { forcePoll: job.status === "running" || job.status === "queued" });
        });
        right.appendChild(viewBtn);
        item.append(left, right);
        container.appendChild(item);
    }


    async function handleClassifierUploadChange(event) {
        const input = event.target;
        const file = input.files && input.files[0];
        input.value = "";
        if (!file) {
            return;
        }
        if (activeElements.classifierPath) {
            activeElements.classifierPath.value = file.name;
        }
        const formData = new FormData();
        formData.append('file', file);
        try {
            const resp = await fetch(`${API_ROOT}/fs/upload_classifier`, {
                method: 'POST',
                body: formData,
            });
            if (!resp.ok) {
                const textBody = await resp.text();
                throw new Error(textBody || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const savedPath = data && data.path ? data.path : null;
            if (savedPath && activeElements.classifierPath) {
                activeElements.classifierPath.value = savedPath;
                setActiveMessage('Classifier uploaded and path updated.', 'success');
            } else if (activeElements.classifierPath) {
                setActiveMessage('Classifier staged locally; enter the server path manually.', 'warn');
            }
        } catch (error) {
            console.error('Classifier upload failed', error);
            setActiveMessage(error.message || String(error), 'error');
        }
    }

    async function handleLabelmapUploadChange(event) {
        const input = event.target;
        const file = input.files && input.files[0];
        input.value = "";
        if (!file) {
            return;
        }
        if (activeElements.labelmapPath) {
            activeElements.labelmapPath.value = file.name;
        }
        const formData = new FormData();
        formData.append('file', file);
        try {
            const resp = await fetch(`${API_ROOT}/fs/upload_labelmap`, {
                method: 'POST',
                body: formData,
            });
            if (!resp.ok) {
                const textBody = await resp.text();
                throw new Error(textBody || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const savedPath = data && data.path ? data.path : null;
            if (savedPath && activeElements.labelmapPath) {
                activeElements.labelmapPath.value = savedPath;
                setActiveMessage('Labelmap uploaded and path updated.', 'success');
            } else if (activeElements.labelmapPath) {
                setActiveMessage('Labelmap staged locally; enter the server path manually.', 'warn');
            }
        } catch (error) {
            console.error('Labelmap upload failed', error);
            setActiveMessage(error.message || String(error), 'error');
        }
    }


    async function chooseOutputDirectory() {
        try {
            const params = new URLSearchParams();
            if (trainingState.outputDirPath) {
                params.set('initial', trainingState.outputDirPath);
            }
            const resp = await fetch(`${API_ROOT}/fs/select_directory${params.toString() ? `?${params.toString()}` : ''}`);
            if (!resp.ok) {
                const textBody = await resp.text();
                let detail = textBody;
                try {
                    detail = JSON.parse(textBody).detail;
                } catch (err) {
                    // ignore JSON parse errors
                }
                if (detail === "tkinter_unavailable") {
                    const fallback = window.prompt('Enter output directory path', trainingState.outputDirPath || '.');
                    if (fallback !== null) {
                        trainingState.outputDirPath = fallback.trim() || '.';
                        if (trainingElements.outputDirSummary) {
                            trainingElements.outputDirSummary.textContent = trainingState.outputDirPath === '.' ? 'Server default (.)' : trainingState.outputDirPath;
                        }
                        setTrainingMessage('Using manual output directory entry.', 'success');
                    }
                    return;
                }
                throw new Error(detail || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const selected = data && Object.prototype.hasOwnProperty.call(data, 'path') ? data.path : null;
            if (!selected) {
                setTrainingMessage('Directory selection cancelled.', null);
                return;
            }
            trainingState.outputDirPath = selected;
            if (trainingElements.outputDirSummary) {
                trainingElements.outputDirSummary.textContent = selected === '.' ? 'Server default (.)' : selected;
            }
            setTrainingMessage('Output directory updated.', 'success');
        } catch (error) {
            console.warn('Directory picker failed', error);
            const fallback = window.prompt('Enter output directory path', trainingState.outputDirPath || '.');
            if (fallback !== null) {
                trainingState.outputDirPath = fallback.trim() || '.';
                if (trainingElements.outputDirSummary) {
                    trainingElements.outputDirSummary.textContent = trainingState.outputDirPath === '.' ? 'Server default (.)' : trainingState.outputDirPath;
                }
                setTrainingMessage('Using manual output directory entry.', 'success');
            }
        }
    }

    async function refreshTrainingHistory() {
        if (!trainingElements.historyContainer) {
            return;
        }
        try {
            const resp = await fetch(`${API_ROOT}/clip/train`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const data = await resp.json();
            trainingElements.historyContainer.innerHTML = "";
            if (!Array.isArray(data) || !data.length) {
                const empty = document.createElement("div");
                empty.textContent = "No training jobs yet.";
                trainingElements.historyContainer.appendChild(empty);
                return;
            }
            data.forEach((job) => renderTrainingHistoryItem(trainingElements.historyContainer, job));
        } catch (error) {
            trainingElements.historyContainer.textContent = `Unable to load history: ${error.message || error}`;
        }
    }

    function renderTrainingStatus(status) {
        trainingState.jobs.set(status.job_id, status);
        const pct = Math.max(0, Math.min(100, Math.round((status.progress || 0) * 100)));
        if (trainingElements.progressFill) {
            trainingElements.progressFill.style.width = `${pct}%`;
        }
        if (trainingElements.cancelButton) {
            const cancellable = status.status === "running" || status.status === "queued";
            trainingElements.cancelButton.disabled = !cancellable;
        }
        if (trainingElements.statusText) {
            const message = status.message ? ` — ${status.message}` : "";
            trainingElements.statusText.textContent = `${status.status}${message}`;
        }
        if (trainingElements.log) {
            const logs = Array.isArray(status.logs) ? status.logs : [];
            const lines = logs.map((entry) => {
                const time = formatTimestamp(entry.timestamp);
                return time ? `[${time}] ${entry.message}` : entry.message;
            });
            trainingElements.log.textContent = lines.join("\n");
        }
        if (trainingElements.summary) {
            if (status.artifacts) {
                const art = status.artifacts;
                const accuracyPct = Number.isFinite(art.accuracy) ? `${(art.accuracy * 100).toFixed(1)}%` : "n/a";
                const iterationsInfo = Number.isFinite(art.iterations_run) ? art.iterations_run : "n/a";
                const convergedInfo = typeof art.converged === "boolean"
                    ? (art.converged ? "yes" : "no")
                    : "n/a";
                const summaryLines = [
                    `Model: ${escapeHtml(art.model_path)}`,
                    `Labelmap: ${escapeHtml(art.labelmap_path)}`,
                    `Accuracy: ${accuracyPct}`,
                    `Train samples: ${art.samples_train}`,
                    `Test samples: ${art.samples_test}`,
                    `Classes seen: ${art.classes_seen}`,
                    `Class weight: ${escapeHtml(art.class_weight || 'none')}`,
                    `Solver: ${escapeHtml(art.solver || 'saga')}`,
                    `Iterations: ${escapeHtml(String(iterationsInfo))}`,
                    `Converged: ${escapeHtml(convergedInfo)}`,
                    `Hard mining: ${art.hard_example_mining ? 'yes' : 'no'}`,
                    `Hard W (misclass): ${escapeHtml(formatNumber(art.hard_mining_misclassified_weight, 2))}`,
                    `Hard W (low conf): ${escapeHtml(formatNumber(art.hard_mining_low_conf_weight, 2))}`,
                    `Low-conf threshold: ${escapeHtml(formatNumber(art.hard_mining_low_conf_threshold, 3))}`,
                    `Margin threshold: ${escapeHtml(formatNumber(art.hard_mining_margin_threshold, 3))}`,
                    `Convergence tol: ${escapeHtml(formatNumber(art.convergence_tol, 6))}`,
                ];
                const summaryHtml = summaryLines.map((line) => `<div>${line}</div>`).join("");
                const perClassHtml = renderPerClassMetrics(art.per_class_metrics);
                const convergenceHtml = renderConvergenceTable(art.convergence_trace);
                trainingElements.summary.innerHTML = summaryHtml + perClassHtml + convergenceHtml;
                trainingState.latestArtifacts = art;
            } else if (status.status !== "succeeded") {
                trainingElements.summary.textContent = "";
            }
        }
        if (status.status === "succeeded" && status.artifacts) {
            setTrainingMessage("Training completed successfully.", "success");
            setActiveMessage("Training completed successfully. Activate it from the CLIP Model tab.", "success");
            stopTrainingPoll();
            if (trainingElements.cancelButton) {
                trainingElements.cancelButton.disabled = true;
            }
        } else if (status.status === "failed") {
            const message = status.error || "Training failed.";
            setTrainingMessage(message, "error");
            setActiveMessage(message, "error");
            stopTrainingPoll();
            if (trainingElements.cancelButton) {
                trainingElements.cancelButton.disabled = true;
            }
        } else if (status.status === "cancelled") {
            setTrainingMessage("Training cancelled.", "warn");
            setActiveMessage("Training cancelled.", "warn");
            stopTrainingPoll();
            if (trainingElements.cancelButton) {
                trainingElements.cancelButton.disabled = true;
            }
        } else if (status.status === "cancelling") {
            setTrainingMessage("Cancellation in progress…", "warn");
            setActiveMessage("Cancellation in progress…", "warn");
            trainingState.pollHandle = setTimeout(() => {
                pollTrainingJob(status.job_id).catch((err) => {
                    console.error("Training poll error", err);
                });
            }, 1500);
        } else if (status.status === "running" || status.status === "queued") {
            trainingState.pollHandle = setTimeout(() => {
                pollTrainingJob(status.job_id).catch((err) => {
                    console.error("Training poll error", err);
                });
            }, 1500);
        }
    }

    async function pollTrainingJob(jobId) {
        if (trainingState.activeJobId !== jobId) {
            return;
        }
        trainingState.pollHandle = null;
        try {
            const resp = await fetch(`${API_ROOT}/clip/train/${jobId}`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const status = await resp.json();
            renderTrainingStatus(status);
        } catch (error) {
            console.warn("Polling training job failed", error);
            setTrainingMessage(`Error polling training job: ${error.message || error}`, "error");
            stopTrainingPoll();
        }
    }

    async function loadTrainingJob(jobId, { forcePoll = false } = {}) {
        trainingState.activeJobId = jobId;
        try {
            const resp = await fetch(`${API_ROOT}/clip/train/${jobId}`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const status = await resp.json();
            renderTrainingStatus(status);
            if (forcePoll || status.status === "running" || status.status === "queued") {
                startTrainingPoll(jobId, true);
            } else {
                stopTrainingPoll();
            }
        } catch (error) {
            setTrainingMessage(`Failed to load job ${jobId}: ${error.message || error}`, "error");
        }
    }

    function gatherTrainingFormData() {
        if (!trainingElements.imagesInput || !trainingElements.labelsInput) {
            throw new Error("Training form not ready");
        }
        const usingNativeImages = Boolean(trainingState.nativeImagesPath);
        const usingNativeLabels = Boolean(trainingState.nativeLabelsPath);
        if (usingNativeImages !== usingNativeLabels) {
            throw new Error("Provide both images and labels folders via server paths or file selection.");
        }
        let imageEntries = [];
        let labelEntries = [];
        if (!usingNativeImages) {
            imageEntries = getStoredEntries("images");
            if (!imageEntries.length) {
                throw new Error("Select an images folder that contains supported image files.");
            }
            labelEntries = getStoredEntries("labels");
            if (!labelEntries.length) {
                throw new Error("Select a labels folder that contains YOLO .txt files.");
            }
        }
        const formData = new FormData();
        if (usingNativeImages) {
            formData.append("images_path_native", trainingState.nativeImagesPath);
            formData.append("labels_path_native", trainingState.nativeLabelsPath);
        } else {
            imageEntries.forEach(({ file, relativePath }) => {
                formData.append("images", file, relativePath || file.name);
            });
            labelEntries.forEach(({ file, relativePath }) => {
                formData.append("labels", file, relativePath || file.name);
            });
        }
        if (trainingElements.labelmapInput && trainingElements.labelmapInput.files.length === 1) {
            formData.append("labelmap", trainingElements.labelmapInput.files[0]);
        } else if (loadedClassList.length) {
            const blob = new Blob([loadedClassList.join("\n")], { type: "text/plain" });
            formData.append("labelmap", blob, "ui_labelmap.txt");
        }
        if (trainingElements.clipBackboneSelect) {
            formData.append("clip_model_name", trainingElements.clipBackboneSelect.value);
        }
        formData.append("output_dir", trainingState.outputDirPath || '.');
        if (trainingElements.modelFilenameInput) {
            formData.append("model_filename", trainingElements.modelFilenameInput.value.trim() || "my_logreg_model.pkl");
        }
        if (trainingElements.labelmapFilenameInput) {
            formData.append("labelmap_filename", trainingElements.labelmapFilenameInput.value.trim() || "my_label_list.pkl");
        }
        if (trainingElements.testSizeInput) {
            formData.append("test_size", trainingElements.testSizeInput.value || "0.2");
        }
        if (trainingElements.randomSeedInput) {
            formData.append("random_seed", trainingElements.randomSeedInput.value || "42");
        }
        if (trainingElements.batchSizeInput) {
            formData.append("batch_size", trainingElements.batchSizeInput.value || "64");
        }
        if (trainingElements.maxIterInput) {
            formData.append("max_iter", trainingElements.maxIterInput.value || "1000");
        }
        if (trainingElements.minPerClassInput) {
            formData.append("min_per_class", trainingElements.minPerClassInput.value || "2");
        }
        if (trainingElements.regCInput) {
            formData.append("C", trainingElements.regCInput.value || "1.0");
        }
        if (trainingElements.classWeightSelect) {
            formData.append("class_weight", trainingElements.classWeightSelect.value || "none");
        }
        if (trainingElements.hardMisWeightInput) {
            formData.append("hard_mis_weight", trainingElements.hardMisWeightInput.value || "3.0");
        }
        if (trainingElements.hardLowConfWeightInput) {
            formData.append("hard_low_conf_weight", trainingElements.hardLowConfWeightInput.value || "2.0");
        }
        if (trainingElements.hardLowConfThresholdInput) {
            formData.append("hard_low_conf_threshold", trainingElements.hardLowConfThresholdInput.value || "0.65");
        }
        if (trainingElements.hardMarginThresholdInput) {
            formData.append("hard_margin_threshold", trainingElements.hardMarginThresholdInput.value || "0.15");
        }
        if (trainingElements.convergenceTolInput) {
            formData.append("convergence_tol", trainingElements.convergenceTolInput.value || "0.0001");
        }
        if (trainingElements.deviceOverrideInput && trainingElements.deviceOverrideInput.value.trim()) {
            formData.append("device_override", trainingElements.deviceOverrideInput.value.trim());
        }
        if (trainingElements.solverSelect) {
            formData.append("solver", trainingElements.solverSelect.value || "saga");
        }
        if (trainingElements.reuseEmbeddingsCheckbox && trainingElements.reuseEmbeddingsCheckbox.checked) {
            formData.append("reuse_embeddings", "true");
        }
        if (trainingElements.hardMiningCheckbox && trainingElements.hardMiningCheckbox.checked) {
            formData.append("hard_example_mining", "true");
        }
        return formData;
    }

    async function handleStartTrainingClick() {
        if (!trainingElements.startButton) {
            return;
        }
        try {
            const formData = gatherTrainingFormData();
            trainingElements.startButton.disabled = true;
            setTrainingMessage("Submitting training job…", null);
            setActiveMessage("Submitting training job…", null);
            if (trainingElements.summary) {
                trainingElements.summary.textContent = "";
            }
            if (trainingElements.log) {
                trainingElements.log.textContent = "";
            }
            if (trainingElements.progressFill) {
                trainingElements.progressFill.style.width = "0%";
            }
            if (trainingElements.statusText) {
                trainingElements.statusText.textContent = "queued";
            }
            if (trainingElements.cancelButton) {
                trainingElements.cancelButton.disabled = true;
            }
            stopTrainingPoll();
            const resp = await fetch(`${API_ROOT}/clip/train`, {
                method: "POST",
                body: formData,
            });
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const jobId = data.job_id;
            if (!jobId) {
                throw new Error("Training job id missing in response.");
            }
            trainingState.latestArtifacts = null;
            setTrainingMessage(`Training job ${jobId} started.`, "success");
            setActiveMessage(`Training job ${jobId} started.`, "success");
            startTrainingPoll(jobId, true);
            refreshTrainingHistory();
            if (trainingElements.cancelButton) {
                trainingElements.cancelButton.disabled = false;
            }
        } catch (error) {
            console.error("Failed to start training", error);
            const msg = error.message || String(error);
            setTrainingMessage(msg, "error");
            setActiveMessage(msg, "error");
        } finally {
            if (trainingElements.startButton) {
                trainingElements.startButton.disabled = false;
            }
        }
    }

    async function cancelActiveTrainingJob() {
        if (!trainingState.activeJobId) {
            setTrainingMessage("No active training job to cancel.", "warn");
            return;
        }
        if (trainingElements.cancelButton) {
            trainingElements.cancelButton.disabled = true;
        }
        try {
            setTrainingMessage("Requesting cancellation…", "warn");
            setActiveMessage("Requesting cancellation…", "warn");
            const resp = await fetch(`${API_ROOT}/clip/train/${trainingState.activeJobId}/cancel`, {
                method: "POST",
            });
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
            }
        } catch (error) {
            console.error("Failed to cancel training", error);
            const msg = error.message || String(error);
            setTrainingMessage(msg, "error");
            setActiveMessage(msg, "error");
            if (trainingElements.cancelButton) {
                trainingElements.cancelButton.disabled = false;
            }
        }
    }

    async function handleActivateLatestModel() {
        const art = trainingState.latestArtifacts;
        if (!art) {
            setActiveMessage("No completed training artifacts available.", "error");
            return;
        }
        try {
            const payload = {
                classifier_path: art.model_path,
                labelmap_path: art.labelmap_path,
                clip_model: art.clip_model || (activeElements.clipSelect ? activeElements.clipSelect.value : null),
            };
            if (activeElements.clipSelect && art.clip_model) {
                activeElements.clipSelect.value = art.clip_model;
            }
            const resp = await fetch(`${API_ROOT}/clip/active_model`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
            }
            setActiveMessage("Activated trained model for labeling.", "success");
            await refreshActiveModelPanel();
            await populateClipBackbones();
        } catch (error) {
            console.error("Failed to activate trained model", error);
            setActiveMessage(`Activation failed: ${error.message || error}`, "error");
        }
    }

    async function handleApplyActiveModel() {
        if (!activeElements.classifierPath || !activeElements.clipSelect) {
            return;
        }
        const payload = {
            classifier_path: activeElements.classifierPath.value.trim() || null,
            labelmap_path: activeElements.labelmapPath ? activeElements.labelmapPath.value.trim() || null : null,
            clip_model: activeElements.clipSelect.value || null,
        };
        if (!payload.classifier_path && !payload.clip_model) {
            setActiveMessage("Provide a classifier path or choose a backbone to apply.", "error");
            return;
        }
        try {
            const resp = await fetch(`${API_ROOT}/clip/active_model`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
            }
            setActiveMessage("Updated active model configuration.", "success");
            await refreshActiveModelPanel();
            await populateClipBackbones();
        } catch (error) {
            console.error("Failed to update active model", error);
            setActiveMessage(`Failed to update active model: ${error.message || error}`, "error");
        }
    }

    function setupTabNavigation() {
        tabElements.labelingButton = document.getElementById("tabLabelingButton");
        tabElements.trainingButton = document.getElementById("tabTrainingButton");
        tabElements.activeButton = document.getElementById("tabActiveButton");
        tabElements.labelingPanel = document.getElementById("tabLabeling");
        tabElements.trainingPanel = document.getElementById("tabTraining");
        tabElements.activePanel = document.getElementById("tabActive");
        if (tabElements.labelingButton) {
            tabElements.labelingButton.addEventListener("click", () => setActiveTab(TAB_LABELING));
        }
        if (tabElements.trainingButton) {
            tabElements.trainingButton.addEventListener("click", () => setActiveTab(TAB_TRAINING));
        }
        if (tabElements.activeButton) {
            tabElements.activeButton.addEventListener("click", () => setActiveTab(TAB_ACTIVE));
        }
        setActiveTab(activeTab);
    }

    function setActiveTab(tabName) {
        const previous = activeTab;
        activeTab = tabName;
        if (tabElements.labelingButton) {
            tabElements.labelingButton.classList.toggle("active", tabName === TAB_LABELING);
        }
        if (tabElements.trainingButton) {
            tabElements.trainingButton.classList.toggle("active", tabName === TAB_TRAINING);
        }
        if (tabElements.activeButton) {
            tabElements.activeButton.classList.toggle("active", tabName === TAB_ACTIVE);
        }
        if (tabElements.labelingPanel) {
            tabElements.labelingPanel.classList.toggle("active", tabName === TAB_LABELING);
        }
        if (tabElements.trainingPanel) {
            tabElements.trainingPanel.classList.toggle("active", tabName === TAB_TRAINING);
        }
        if (tabElements.activePanel) {
            tabElements.activePanel.classList.toggle("active", tabName === TAB_ACTIVE);
        }
        if (tabName === TAB_TRAINING && previous !== TAB_TRAINING) {
            initializeTrainingUi();
            refreshTrainingHistory();
            populateClipBackbones();
            if (trainingState.activeJobId) {
                loadTrainingJob(trainingState.activeJobId, { forcePoll: true });
            }
        }
        if (tabName === TAB_ACTIVE && previous !== TAB_ACTIVE) {
            initializeActiveModelUi();
            populateClipBackbones();
            refreshActiveModelPanel();
        }
        if (tabName === TAB_LABELING) {
            ensureCanvasDimensions();
            if (currentImage && fittedZoom) {
                fitZoom(currentImage, { preservePan: true });
            }
        }
        if (tabName !== TAB_TRAINING && previous === TAB_TRAINING) {
            stopTrainingPoll();
        }
    }

    function initializeTrainingUi() {
        if (trainingUiInitialized) {
            return;
        }
        trainingUiInitialized = true;
        trainingElements.clipBackboneSelect = document.getElementById("clipBackboneSelect");
        trainingElements.solverSelect = document.getElementById("trainSolver");
        trainingElements.imagesInput = document.getElementById("trainImages");
        trainingElements.imagesBtn = document.getElementById("trainImagesBtn");
        trainingElements.imagesSummary = document.getElementById("trainImagesSummary");
        trainingElements.labelsInput = document.getElementById("trainLabels");
        trainingElements.labelsBtn = document.getElementById("trainLabelsBtn");
        trainingElements.labelsSummary = document.getElementById("trainLabelsSummary");
        trainingElements.labelmapInput = document.getElementById("trainLabelmap");
        trainingElements.labelmapSummary = document.getElementById("trainLabelmapSummary");
        trainingElements.outputDirBtn = document.getElementById("trainOutputDirBtn");
        trainingElements.outputDirSummary = document.getElementById("trainOutputDirSummary");
        trainingElements.modelFilenameInput = document.getElementById("trainModelFilename");
        trainingElements.labelmapFilenameInput = document.getElementById("trainLabelmapFilename");
        trainingElements.testSizeInput = document.getElementById("trainTestSize");
        trainingElements.randomSeedInput = document.getElementById("trainRandomSeed");
        trainingElements.batchSizeInput = document.getElementById("trainBatchSize");
        trainingElements.maxIterInput = document.getElementById("trainMaxIter");
        trainingElements.minPerClassInput = document.getElementById("trainMinPerClass");
        trainingElements.regCInput = document.getElementById("trainRegC");
        trainingElements.classWeightSelect = document.getElementById("trainClassWeight");
        trainingElements.deviceOverrideInput = document.getElementById("trainDeviceOverride");
        trainingElements.hardMisWeightInput = document.getElementById("trainHardMisWeight");
        trainingElements.hardLowConfWeightInput = document.getElementById("trainHardLowConfWeight");
        trainingElements.hardLowConfThresholdInput = document.getElementById("trainHardLowConfThreshold");
        trainingElements.hardMarginThresholdInput = document.getElementById("trainHardMarginThreshold");
        trainingElements.convergenceTolInput = document.getElementById("trainConvergenceTol");
        trainingElements.startButton = document.getElementById("startTrainingBtn");
        trainingElements.cancelButton = document.getElementById("cancelTrainingBtn");
        trainingElements.progressFill = document.getElementById("trainingProgressFill");
        trainingElements.statusText = document.getElementById("trainingStatusText");
        trainingElements.message = document.getElementById("trainingMessage");
        trainingElements.summary = document.getElementById("trainingSummary");
        trainingElements.log = document.getElementById("trainingLog");
        trainingElements.historyContainer = document.getElementById("trainingHistory");
        trainingElements.reuseEmbeddingsCheckbox = document.getElementById("trainReuseEmbeddings");
        trainingElements.hardMiningCheckbox = document.getElementById("trainHardMining");

        if (trainingElements.imagesBtn) {
            trainingElements.imagesBtn.addEventListener("click", () => chooseTrainingFolder("images"));
        }
        if (trainingElements.imagesInput) {
            trainingElements.imagesInput.addEventListener("change", handleImagesInputChange);
        }
        if (trainingElements.labelsBtn) {
            trainingElements.labelsBtn.addEventListener("click", () => chooseTrainingFolder("labels"));
        }
        if (trainingElements.labelsInput) {
            trainingElements.labelsInput.addEventListener("change", handleLabelsInputChange);
        }
        if (trainingElements.labelmapInput) {
            trainingElements.labelmapInput.addEventListener("change", () => updateFileSummary(trainingElements.labelmapInput, trainingElements.labelmapSummary, { emptyText: "Optional" }));
        }
        if (trainingElements.outputDirBtn) {
            trainingElements.outputDirBtn.addEventListener("click", () => {
                chooseOutputDirectory().catch((err) => console.error("Directory picker error", err));
            });
        }
        if (trainingElements.startButton) {
            trainingElements.startButton.addEventListener("click", () => {
                handleStartTrainingClick().catch((err) => {
                    console.error("Training submit error", err);
                });
            });
        }
        if (trainingElements.cancelButton) {
            trainingElements.cancelButton.addEventListener("click", () => {
                cancelActiveTrainingJob().catch((err) => {
                    console.error("Cancel training error", err);
                });
            });
        }

        updateFileSummary(trainingElements.imagesInput, trainingElements.imagesSummary, { emptyText: "No folder selected", mode: "folder", allowedExts: IMAGE_EXTENSIONS });
        updateFileSummary(trainingElements.labelsInput, trainingElements.labelsSummary, { emptyText: "No folder selected", mode: "folder", allowedExts: LABEL_EXTENSIONS });
        updateFileSummary(trainingElements.labelmapInput, trainingElements.labelmapSummary, { emptyText: "Optional" });
        if (trainingElements.outputDirSummary) {
            trainingElements.outputDirSummary.textContent = trainingState.outputDirPath && trainingState.outputDirPath !== '.' ? trainingState.outputDirPath : 'Server default (.)';
        }

        populateClipBackbones();
    }

    function initializeActiveModelUi() {
        if (activeUiInitialized) {
            return;
        }
        activeUiInitialized = true;
        activeElements.message = document.getElementById("activeMessage");
        activeElements.info = document.getElementById("activeModelInfo");
        activeElements.clipSelect = document.getElementById("activeClipSelect");
        activeElements.classifierPath = document.getElementById("activeClassifierPath");
        activeElements.classifierUpload = document.getElementById("activeClassifierUpload");
        activeElements.classifierBrowse = document.getElementById("activeClassifierBrowse");
        activeElements.labelmapPath = document.getElementById("activeLabelmapPath");
        activeElements.labelmapUpload = document.getElementById("activeLabelmapUpload");
        activeElements.labelmapBrowse = document.getElementById("activeLabelmapBrowse");
        activeElements.activateLatestButton = document.getElementById("activateLatestModelBtn");
        activeElements.applyButton = document.getElementById("applyActiveModelBtn");
        activeElements.refreshButton = document.getElementById("refreshActiveModelBtn");

        if (activeElements.activateLatestButton) {
            activeElements.activateLatestButton.addEventListener("click", () => {
                handleActivateLatestModel().catch((err) => {
                    console.error("Activate latest model error", err);
                });
            });
        }
        if (activeElements.applyButton) {
            activeElements.applyButton.addEventListener("click", () => {
                handleApplyActiveModel().catch((err) => {
                    console.error("Apply active model error", err);
                });
            });
        }
        if (activeElements.refreshButton) {
            activeElements.refreshButton.addEventListener("click", () => {
                refreshActiveModelPanel();
                populateClipBackbones();
            });
        }
        if (activeElements.classifierBrowse && activeElements.classifierUpload) {
            activeElements.classifierBrowse.addEventListener("click", (event) => {
                event.preventDefault();
                activeElements.classifierUpload.click();
            });
        }
        if (activeElements.classifierUpload) {
            activeElements.classifierUpload.addEventListener("change", (event) => {
                handleClassifierUploadChange(event).catch((err) => console.error('Classifier upload error', err));
            });
        }
        if (activeElements.labelmapBrowse && activeElements.labelmapUpload) {
            activeElements.labelmapBrowse.addEventListener("click", (event) => {
                event.preventDefault();
                activeElements.labelmapUpload.click();
            });
        }
        if (activeElements.labelmapUpload) {
            activeElements.labelmapUpload.addEventListener("change", (event) => {
                handleLabelmapUploadChange(event).catch((err) => console.error('Labelmap upload error', err));
            });
        }
    }


    function removePendingBbox(context) {
        const info = context || multiPointPendingBboxInfo;
        if (!info) {
            return;
        }
        const { uuid, imageName } = info;
        if (uuid) {
            delete pendingApiBboxes[uuid];
        }
        if (currentBbox && currentBbox.bbox && currentBbox.bbox.uuid === uuid) {
            currentBbox = null;
        }
        if (imageName && bboxes[imageName]) {
            const classBuckets = bboxes[imageName];
            for (const className of Object.keys(classBuckets)) {
                const bucket = classBuckets[className];
                const idx = bucket.findIndex((bbox) => bbox.uuid === uuid);
                if (idx !== -1) {
                    bucket.splice(idx, 1);
                    break;
                }
            }
        }
        if (!context || (multiPointPendingBboxInfo && multiPointPendingBboxInfo.uuid === uuid && multiPointPendingBboxInfo.imageName === imageName)) {
            multiPointPendingBboxInfo = null;
        }
    }

    function clearMultiPointAnnotations() {
        multiPointPoints = [];
    }

    function cancelPendingMultiPoint({ clearMarkers = false, removePendingBbox: removePendingBboxFlag = false } = {}) {
        multiPointPending = false;
        multiPointPendingToken = null;
        multiPointPendingBboxInfo = null;
        if (removePendingBboxFlag) {
            removePendingBbox();
        }
        if (multiPointQueue.length > 0 && removePendingBboxFlag) {
            while (multiPointQueue.length) {
                const job = multiPointQueue.shift();
                if (job?.placeholderContext) {
                    removePendingBbox(job.placeholderContext);
                }
            }
        } else {
            multiPointQueue.length = 0;
        }
        if (clearMarkers) {
            clearMultiPointAnnotations();
        }
        multiPointWaitingForPreload = false;
    }

    function setButtonDisabled(button, disabled) {
        if (!button) {
            return;
        }
        if (disabled) {
            button.classList.add("button-disabled");
            button.setAttribute("aria-disabled", "true");
            button.setAttribute("tabindex", "-1");
        } else {
            button.classList.remove("button-disabled");
            button.removeAttribute("aria-disabled");
            button.setAttribute("tabindex", "0");
        }
    }

    function registerFileLabel(label, input) {
        if (!label || !input) {
            return;
        }
        label.addEventListener("click", (event) => {
            if (input.disabled) {
                event.preventDefault();
                return;
            }
            event.preventDefault();
            input.value = "";
            input.click();
        });
        label.addEventListener("keydown", (event) => {
            if ((event.key === "Enter" || event.key === " ") && !input.disabled) {
                event.preventDefault();
                input.value = "";
                input.click();
            }
        });
    }

    function setSamStatus(text, { variant = null, duration = 4000 } = {}) {
        if (!samStatusEl) {
            samStatusMessageToken++;
            return samStatusMessageToken;
        }
        if (samStatusTimer) {
            clearTimeout(samStatusTimer);
            samStatusTimer = null;
        }
        const token = ++samStatusMessageToken;
        samStatusEl.textContent = text || "";
        samStatusEl.classList.remove("warn", "error", "success");
        if (variant) {
            samStatusEl.classList.add(variant);
        }
        if (text && duration !== 0) {
            const timeout = typeof duration === "number" ? duration : 4000;
            samStatusTimer = setTimeout(() => {
                if (samStatusMessageToken !== token) {
                    return;
                }
                samStatusEl.textContent = "";
                samStatusEl.classList.remove("warn", "error", "success");
                samStatusTimer = null;
                hideSamPreloadProgress();
            }, timeout);
        } else if (!text) {
            hideSamPreloadProgress();
        }
        return token;
    }

    function showSamPreloadProgress() {
        if (!samStatusProgressEl) {
            return;
        }
        samStatusProgressEl.classList.add("active");
        samStatusProgressEl.setAttribute("aria-hidden", "false");
    }

    function hideSamPreloadProgress() {
        if (!samStatusProgressEl) {
            return;
        }
        samStatusProgressEl.classList.remove("active");
        samStatusProgressEl.setAttribute("aria-hidden", "true");
    }

    function beginSamActionStatus(message, { variant = "info" } = {}) {
        const token = setSamStatus(message, { variant, duration: 0 });
        showSamPreloadProgress();
        return token;
    }

    function endSamActionStatus(token, options = {}) {
        if (samStatusMessageToken !== token) {
            return;
        }
        const { message = "", variant = null, duration = 0 } = options;
        if (message) {
            setSamStatus(message, { variant, duration });
        } else {
            setSamStatus("", { duration: 0 });
        }
        hideSamPreloadProgress();
    }

    function registerSamPreloadWatcher(imageName, variant) {
        return new Promise((resolve) => {
            const key = getSamTokenKey(imageName, variant);
            const entry = { resolver: resolve, timeoutId: null };
            const bucket = samPreloadWatchers.get(key) || [];
            entry.timeoutId = window.setTimeout(() => {
                const current = samPreloadWatchers.get(key);
                if (current) {
                    const idx = current.indexOf(entry);
                    if (idx !== -1) {
                        current.splice(idx, 1);
                        if (current.length === 0) {
                            samPreloadWatchers.delete(key);
                        }
                    }
                }
                resolve(null);
            }, SAM_PRELOAD_WAIT_TIMEOUT_MS);
            bucket.push(entry);
            samPreloadWatchers.set(key, bucket);
        });
    }

    function notifySamPreloadWatchers(imageName, variant) {
        if (!imageName) {
            return;
        }
        const key = getSamTokenKey(imageName, variant);
        const listeners = samPreloadWatchers.get(key);
        if (!listeners) {
            return;
        }
        samPreloadWatchers.delete(key);
        const token = getSamToken(imageName, variant) || null;
        listeners.forEach(({ resolver, timeoutId }) => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
            try {
                resolver(token);
            } catch (error) {
                console.warn("SAM preload watcher resolution failed", error);
            }
        });
    }

    function waitForSamPreloadIfActive(imageName, variant) {
        if (!imageName) {
            return Promise.resolve(getSamToken(imageName, variant) || null);
        }
        if (!currentImage || currentImage.name !== imageName) {
            return Promise.resolve(getSamToken(imageName, variant) || null);
        }
        const effectiveVariant = variant || samVariant;
        const immediateToken = getSamToken(imageName, effectiveVariant);
        if (immediateToken) {
            return Promise.resolve(immediateToken);
        }
        if (!samPreloadEnabled) {
            return Promise.resolve(null);
        }
        if (!samPreloadCurrentImageName) {
            return Promise.resolve(null);
        }
        if (samPreloadCurrentImageName !== imageName) {
            return Promise.resolve(null);
        }
        const activeVariant = samPreloadCurrentVariant || samVariant;
        const watchVariant = activeVariant || effectiveVariant;

        return (async () => {
            let attempts = 0;
            while (attempts < 3) {
                const awaitedToken = await registerSamPreloadWatcher(imageName, watchVariant);
                const candidateVariant = awaitedToken ? watchVariant : (samPreloadCurrentVariant || samVariant || effectiveVariant);
                const candidateToken = awaitedToken || getSamToken(imageName, candidateVariant);
                if (candidateToken) {
                    return candidateToken;
                }
                if (!samPreloadEnabled || !samPreloadCurrentImageName || samPreloadCurrentImageName !== imageName) {
                    break;
                }
                attempts += 1;
            }
            return getSamToken(imageName, effectiveVariant) || null;
        })();
    }

    function resetSamPreloadState() {
        const finishedImage = samPreloadCurrentImageName;
        const finishedVariant = samPreloadCurrentVariant;
        samPreloadCurrentImageName = null;
        samPreloadCurrentVariant = null;
        return { finishedImage, finishedVariant };
    }

    function resolveSamPreloadWaiters(imageName, variant) {
        if (imageName) {
            const variantsToNotify = new Set();
            if (variant) {
                variantsToNotify.add(variant);
            }
            variantsToNotify.add(samVariant);
            const imageKeySuffix = `::${imageName || ""}`;
            samPreloadWatchers.forEach((_, key) => {
                if (key.endsWith(imageKeySuffix)) {
                    const variantKey = key.slice(0, -imageKeySuffix.length) || "sam1";
                    variantsToNotify.add(variantKey);
                }
            });
            variantsToNotify.forEach((variantKey) => {
                notifySamPreloadWatchers(imageName, variantKey);
            });
        }
        resumeMultiPointQueueIfIdle();
    }

    function focusMultiPointPlaceholder(job) {
        if (!job?.placeholderContext) {
            currentBbox = null;
            return;
        }
        const { uuid, imageName } = job.placeholderContext;
        if (!currentImage || currentImage.name !== imageName) {
            currentBbox = null;
            return;
        }
        const classBuckets = bboxes[imageName];
        if (!classBuckets) {
            currentBbox = null;
            return;
        }
        for (const className of Object.keys(classBuckets)) {
            const bucket = classBuckets[className];
            const idx = bucket.findIndex((bbox) => bbox.uuid === uuid);
            if (idx !== -1) {
                const bbox = bucket[idx];
                currentBbox = {
                    bbox,
                    index: idx,
                    originalX: bbox.x,
                    originalY: bbox.y,
                    originalWidth: bbox.width,
                    originalHeight: bbox.height,
                    moving: false,
                    resizing: null,
                };
                return;
            }
        }
        currentBbox = null;
    }

    function scheduleMultiPointProcessing() {
        Promise.resolve()
            .then(() => processNextMultiPointJob())
            .catch((err) => {
                console.error("Multi-point queue error", err);
            });
    }

    function resumeMultiPointQueueIfIdle() {
        if (!multiPointWaitingForPreload) {
            return;
        }
        if (samPreloadCurrentImageName) {
            return;
        }
        if (multiPointPending) {
            return;
        }
        multiPointWaitingForPreload = false;
        if (multiPointQueue.length) {
            scheduleMultiPointProcessing();
        }
    }

    function enqueueMultiPointJob(job) {
        multiPointQueue.push(job);
        scheduleMultiPointProcessing();
    }

    async function processNextMultiPointJob() {
        if (multiPointPending) {
            return;
        }
        const job = multiPointQueue.shift();
        if (!job) {
            return;
        }
        if (samPreloadCurrentImageName && job.imageName && job.imageName === samPreloadCurrentImageName) {
            multiPointQueue.unshift(job);
            multiPointWaitingForPreload = true;
            return;
        }
        multiPointWaitingForPreload = false;
        if (!job.imageName || !currentImage || currentImage.name !== job.imageName) {
            if (job.placeholderContext) {
                removePendingBbox(job.placeholderContext);
            }
            return processNextMultiPointJob();
        }
        multiPointPending = true;
        multiPointPendingToken = job.requestToken;
        multiPointPendingBboxInfo = job.placeholderContext ? { ...job.placeholderContext } : null;
        focusMultiPointPlaceholder(job);
        const jobHandle = registerSamJob({
            type: job.auto ? "sam-multi-auto" : "sam-multi",
            imageName: job.imageName,
            cleanup: () => {
                if (job.placeholderContext) {
                    removePendingBbox(job.placeholderContext);
                }
            },
        });
        try {
            if (job.auto) {
                await sam2PointMultiAutoPrompt(job, jobHandle);
            } else {
                await sam2PointMultiPrompt(job, jobHandle);
            }
        } finally {
            completeSamJob(jobHandle.id);
            if (multiPointPendingToken === job.requestToken) {
                multiPointPending = false;
                multiPointPendingToken = null;
                multiPointPendingBboxInfo = null;
            }
            if (multiPointQueue.length) {
                scheduleMultiPointProcessing();
            }
        }
    }

    function getSamTokenKey(imageName, variant) {
        return `${variant || "sam1"}::${imageName || ""}`;
    }

    function rememberSamToken(imageName, variant, token) {
        if (!imageName || !token) {
            return;
        }
        samTokenCache.set(getSamTokenKey(imageName, variant), {
            token,
            timestamp: Date.now(),
        });
    }

    function forgetSamToken(imageName, variant) {
        if (!imageName) {
            return;
        }
        samTokenCache.delete(getSamTokenKey(imageName, variant));
    }

    function getSamToken(imageName, variant) {
        const entry = samTokenCache.get(getSamTokenKey(imageName, variant));
        return entry ? entry.token : null;
    }

    function getCurrentSamToken(variantOverride = null) {
        if (!currentImage) {
            return null;
        }
        const variant = variantOverride ?? samVariant;
        return getSamToken(currentImage.name, variant);
    }

    function registerSamJob({ type, imageName, cleanup }) {
        const jobId = ++samJobSequence;
        const record = {
            id: jobId,
            type: type || "sam",
            imageName: imageName || null,
            version: samCancelVersion,
            cleanup: typeof cleanup === "function" ? cleanup : null,
        };
        samActiveJobs.set(jobId, record);
        return { id: jobId, version: record.version };
    }

    function completeSamJob(jobId) {
        samActiveJobs.delete(jobId);
    }

    function isSamJobActive(jobHandle) {
        if (!jobHandle) {
            return false;
        }
        const record = samActiveJobs.get(jobHandle.id);
        return Boolean(record) && record.version === samCancelVersion;
    }

    function cancelAllSamJobs({ reason = "cancelled", imageName = null, announce = true } = {}) {
        if (samActiveJobs.size === 0) {
            return { count: 0, message: null };
        }
        const jobs = Array.from(samActiveJobs.values());
        samActiveJobs.clear();
        samCancelVersion++;
        const count = jobs.length;
        const label = imageName ? ` from ${imageName}` : "";
        const reasonText = reason ? ` (${reason})` : "";
        const message = `Canceled ${count} SAM job${count === 1 ? "" : "s"}${label}${reasonText}`;
        jobs.forEach((job) => {
            if (typeof job.cleanup === "function") {
                try {
                    job.cleanup();
                } catch (error) {
                    console.warn("SAM job cleanup failed", error);
                }
            }
        });
        if (announce && count > 0) {
            setSamStatus(message, { variant: "warn", duration: 5000 });
        }
        return { count, message };
    }

    function updateAutoModeState(checked) {
        autoMode = !!checked;
        if (autoModeCheckbox) {
            autoModeCheckbox.checked = autoMode;
        }
        samAutoMode = samMode && autoMode;
        samPointAutoMode = pointMode && autoMode;
        samMultiPointAutoMode = multiPointMode && autoMode;
        console.log(
            "Auto class =>",
            autoMode,
            "samAutoMode =>",
            samAutoMode,
            "samPointAutoMode =>",
            samPointAutoMode,
            "samMultiPointAutoMode =>",
            samMultiPointAutoMode,
        );
    }

    function updatePointModeState(checked) {
        const enablePointMode = samMode && !!checked;
        pointMode = enablePointMode;
        if (pointModeCheckbox) {
            pointModeCheckbox.checked = pointMode;
            pointModeCheckbox.disabled = !samMode;
        }
        if (pointMode && multiPointMode) {
            updateMultiPointState(false);
        }
        samPointAutoMode = pointMode && autoMode;
        console.log("Point mode =>", pointMode, "samPointAutoMode =>", samPointAutoMode);
    }

    function cancelSamPreload() {
        samPreloadToken++;
        if (samPreloadAbortController) {
            samPreloadAbortController.abort();
            samPreloadAbortController = null;
        }
        if (samPreloadTimer) {
            clearTimeout(samPreloadTimer);
            samPreloadTimer = null;
        }
        const label = samPreloadCurrentImageName;
        if (label) {
            setSamStatus(`Canceled SAM preload for ${label}`, { variant: "warn", duration: 2500 });
        }
        const { finishedImage, finishedVariant } = resetSamPreloadState();
        hideSamPreloadProgress();
        resolveSamPreloadWaiters(finishedImage, finishedVariant);
    }

    function updateSamPreloadState(checked) {
        samPreloadEnabled = !!checked;
        if (samPreloadCheckbox) {
            samPreloadCheckbox.checked = samPreloadEnabled;
        }
        if (!samPreloadEnabled) {
            samPreloadLastKey = null;
            cancelSamPreload();
            return;
        }
        if (currentImage && currentImage.object) {
            scheduleSamPreload({ force: true, immediate: true });
        }
    }

    function scheduleSamPreload(options = {}) {
        if (!samPreloadEnabled || !currentImage || !currentImage.object) {
            hideSamPreloadProgress();
            return;
        }
        if (samPreloadTimer) {
            clearTimeout(samPreloadTimer);
        }
        const delay = typeof options.delayMs === "number"
            ? Math.max(0, options.delayMs)
            : (options.immediate ? 0 : SAM_PRELOAD_DEBOUNCE_MS);
        const targetImage = currentImage;
        const targetVersion = targetImage._loadVersion || 0;
        const generation = ++samPreloadGeneration;
        const requestOptions = {
            force: Boolean(options.force),
            messagePrefix: options.messagePrefix || null,
            imageRef: targetImage,
            version: targetVersion,
            queuedAt: Date.now(),
            generation,
        };
        samPreloadTimer = setTimeout(() => {
            samPreloadTimer = null;
            executeSamPreload(requestOptions).catch((err) => {
                console.warn("SAM preload error", err);
            });
        }, delay);
    }

    async function executeSamPreload(options) {
        const startTime = Date.now();
        const elapsed = startTime - (options.queuedAt || startTime);
        if (elapsed > 3000) {
            return;
        }
        if (options.generation && options.generation < samPreloadGeneration) {
            hideSamPreloadProgress();
            resumeMultiPointQueueIfIdle();
            return;
        }
        if (!samPreloadEnabled) {
            hideSamPreloadProgress();
            resumeMultiPointQueueIfIdle();
            return;
        }
        const activeImage = currentImage;
        if (!activeImage || !options || activeImage !== options.imageRef || (activeImage._loadVersion || 0) !== options.version || !activeImage.object) {
            const { finishedImage, finishedVariant } = resetSamPreloadState();
            hideSamPreloadProgress();
            resolveSamPreloadWaiters(finishedImage, finishedVariant);
            return;
        }

        const variantSnapshot = samVariant;
        const imageSnapshot = activeImage;
        const imageKey = `${imageSnapshot.name || ""}::${variantSnapshot}`;
        const cachedToken = getSamToken(imageSnapshot.name, variantSnapshot);
        const useTokenOnly = !options.force && Boolean(cachedToken);
        samPreloadLastKey = imageKey;

        const newToken = ++samPreloadToken;
        samPreloadLastKey = imageKey;
        let canceledLabel = null;
        if (samPreloadAbortController) {
            samPreloadAbortController.abort();
            canceledLabel = samPreloadCurrentImageName;
        }
        const controller = new AbortController();
        samPreloadAbortController = controller;
        const preloadLabel = imageSnapshot.name || "(unnamed)";
        samPreloadCurrentImageName = preloadLabel;
        samPreloadCurrentVariant = variantSnapshot;
        const prefixPieces = [];
        if (options.messagePrefix) {
            prefixPieces.push(options.messagePrefix);
        }
        if (canceledLabel) {
            prefixPieces.push(`Canceled SAM preload for ${canceledLabel}`);
        }
        const prefixText = prefixPieces.length ? `${prefixPieces.join(" — ")} — ` : "";
        setSamStatus(`${prefixText}Preloading SAM: ${preloadLabel}`, { variant: "info", duration: 0 });
        showSamPreloadProgress();

        try {
            const requestBody = { sam_variant: variantSnapshot };
            let tokenUsed = false;
            if (useTokenOnly) {
                requestBody.image_token = cachedToken;
                tokenUsed = true;
            } else {
                let base64Img;
                if (imageSnapshot.dataUrl && imageSnapshot.dataUrl.includes(',')) {
                    base64Img = imageSnapshot.dataUrl.split(',')[1];
                } else {
                    base64Img = await extractBase64ForImage(imageSnapshot);
                }
                if (newToken !== samPreloadToken) {
                    hideSamPreloadProgress();
                    resumeMultiPointQueueIfIdle();
                    return;
                }
                requestBody.image_base64 = base64Img;
            }
            if (options.generation) {
                requestBody.preload_generation = options.generation;
            }

            let resp = await fetch(`${API_ROOT}/sam_preload`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody),
                signal: controller.signal,
            });

            if (resp.status === 409) {
                const { finishedImage, finishedVariant } = resetSamPreloadState();
                hideSamPreloadProgress();
                resolveSamPreloadWaiters(finishedImage, finishedVariant);
                return;
            }

            if (tokenUsed && resp.status === 404) {
                forgetSamToken(imageSnapshot.name, variantSnapshot);
                let base64Img;
                if (imageSnapshot.dataUrl && imageSnapshot.dataUrl.includes(',')) {
                    base64Img = imageSnapshot.dataUrl.split(',')[1];
                } else {
                    base64Img = await extractBase64ForImage(imageSnapshot);
                }
                if (newToken !== samPreloadToken) {
                    hideSamPreloadProgress();
                    resumeMultiPointQueueIfIdle();
                    return;
                }
                resp = await fetch(`${API_ROOT}/sam_preload`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        image_base64: base64Img,
                        sam_variant: variantSnapshot,
                        preload_generation: options.generation || null,
                    }),
                    signal: controller.signal,
                });
                if (resp.status === 409) {
                    const { finishedImage, finishedVariant } = resetSamPreloadState();
                    hideSamPreloadProgress();
                    resolveSamPreloadWaiters(finishedImage, finishedVariant);
                    return;
                }
            }

            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
            }
            const result = await resp.json();
            if (result?.status === "superseded") {
                const { finishedImage, finishedVariant } = resetSamPreloadState();
                hideSamPreloadProgress();
                resolveSamPreloadWaiters(finishedImage, finishedVariant);
                return;
            }
            if (newToken === samPreloadToken) {
                if (result?.token && imageSnapshot?.name) {
                    rememberSamToken(imageSnapshot.name, variantSnapshot, result.token);
                }
                const { finishedImage, finishedVariant } = resetSamPreloadState();
                setSamStatus(`Image ${preloadLabel} loaded in SAM`, { variant: "success", duration: 3000 });
                hideSamPreloadProgress();
                resolveSamPreloadWaiters(finishedImage, finishedVariant);
            }
        } catch (error) {
            if (error && error.name === "AbortError") {
                hideSamPreloadProgress();
                resumeMultiPointQueueIfIdle();
                return;
            }
            console.warn("SAM preload failed", error);
            if (imageKey === samPreloadLastKey) {
                samPreloadLastKey = null;
            }
            const { finishedImage, finishedVariant } = resetSamPreloadState();
            if (imageSnapshot?.name) {
                forgetSamToken(imageSnapshot.name, variantSnapshot);
            }
            const detail = error && error.message ? error.message : String(error);
            setSamStatus(`SAM preload failed for ${preloadLabel}: ${detail}`, { variant: "error", duration: 6000 });
            hideSamPreloadProgress();
            resolveSamPreloadWaiters(finishedImage, finishedVariant);
        } finally {
            if (samPreloadAbortController === controller) {
                samPreloadAbortController = null;
            }
        }
    }

    function updateMultiPointState(checked, options = {}) {
        const enableMultiPoint = samMode && !!checked;
        const { preservePoints = false } = options;
        multiPointMode = enableMultiPoint;
        if (multiPointModeCheckbox) {
            multiPointModeCheckbox.checked = multiPointMode;
            multiPointModeCheckbox.disabled = !samMode;
        }
        if (multiPointMode && pointMode) {
            updatePointModeState(false);
        }
        if (!multiPointMode) {
            cancelPendingMultiPoint({ clearMarkers: !preservePoints, removePendingBbox: true });
        }
        samMultiPointAutoMode = multiPointMode && autoMode;
        console.log("Multi-point mode =>", multiPointMode, "samMultiPointAutoMode =>", samMultiPointAutoMode);
    }

    function updateSamModeState(checked, options = {}) {
        const { preservePoints = false } = options;
        samMode = !!checked;
        if (samModeCheckbox) {
            samModeCheckbox.checked = samMode;
        }
        samAutoMode = samMode && autoMode;
        if (!samMode) {
            updatePointModeState(false);
            updateMultiPointState(false, { preservePoints });
            cancelAllSamJobs({ reason: "SAM off", imageName: currentImage ? currentImage.name : null });
        } else {
            if (pointModeCheckbox) {
                pointModeCheckbox.disabled = false;
            }
            if (multiPointModeCheckbox) {
                multiPointModeCheckbox.disabled = false;
            }
        }
        console.log("SAM mode =>", samMode, "samAutoMode =>", samAutoMode);
    }

    document.addEventListener("DOMContentLoaded", () => {
        autoModeCheckbox = document.getElementById("autoMode");
        samModeCheckbox = document.getElementById("samMode");
        pointModeCheckbox = document.getElementById("pointMode");
        multiPointModeCheckbox = document.getElementById("multiPointMode");
        samVariantSelect = document.getElementById("samVariant");
        samPreloadCheckbox = document.getElementById("samPreload");
        imagesSelectButton = document.getElementById("imagesSelect");
        classesSelectButton = document.getElementById("classesSelect");
        bboxesSelectButton = document.getElementById("bboxesSelect");
        samStatusEl = document.getElementById("samStatus");
        samStatusProgressEl = document.getElementById("samStatusProgress");

        registerFileLabel(imagesSelectButton, document.getElementById("images"));
        registerFileLabel(classesSelectButton, document.getElementById("classes"));
        registerFileLabel(bboxesSelectButton, document.getElementById("bboxes"));
        hideSamPreloadProgress();

        if (autoModeCheckbox) {
            autoModeCheckbox.addEventListener("change", () => {
                updateAutoModeState(autoModeCheckbox.checked);
            });
        }

        if (samModeCheckbox) {
            samModeCheckbox.addEventListener("change", () => {
                updateSamModeState(samModeCheckbox.checked);
            });
        }

        if (pointModeCheckbox) {
            pointModeCheckbox.addEventListener("change", () => {
                updatePointModeState(pointModeCheckbox.checked);
            });
        }

        if (multiPointModeCheckbox) {
            multiPointModeCheckbox.addEventListener("change", () => {
                updateMultiPointState(multiPointModeCheckbox.checked);
            });
        }

        if (samPreloadCheckbox) {
            samPreloadCheckbox.addEventListener("change", () => {
                updateSamPreloadState(samPreloadCheckbox.checked);
            });
        }

        if (samVariantSelect) {
            samVariant = samVariantSelect.value || "sam1";
            samVariantSelect.addEventListener("change", () => {
                samVariant = samVariantSelect.value || "sam1";
                console.log("SAM variant =>", samVariant);
                samPreloadLastKey = null;
                if (samPreloadCurrentImageName) {
                    samPreloadCurrentVariant = samVariant;
                }
                if (samPreloadEnabled && currentImage && currentImage.object) {
                    scheduleSamPreload({ force: true, immediate: true });
                }
            });
        }

        updateSamModeState(Boolean(samModeCheckbox?.checked));
        updateAutoModeState(Boolean(autoModeCheckbox?.checked));
        updatePointModeState(Boolean(pointModeCheckbox?.checked));
        updateMultiPointState(Boolean(multiPointModeCheckbox?.checked));
        updateSamPreloadState(Boolean(samPreloadCheckbox?.checked));
    });

    // Helper that extracts base64 from currentImage
    async function extractBase64Image() {
        const offCan = document.createElement("canvas");
        offCan.width = currentImage.width;
        offCan.height = currentImage.height;
        const ctx = offCan.getContext("2d");
        ctx.drawImage(currentImage.object, 0, 0);
        const dataUrl = offCan.toDataURL("image/jpeg");
        return dataUrl.split(",")[1];
    }

    async function buildSamImagePayload({ forceBase64 = false, variantOverride = null, preferredToken = null } = {}) {
        const token = preferredToken ?? getCurrentSamToken(variantOverride);
        if (token && !forceBase64) {
            return { image_token: token };
        }
        const base64Img = await extractBase64Image();
        const payload = { image_base64: base64Img };
        if (token) {
            payload.image_token = token;
        }
        return payload;
    }

    async function postSamEndpoint(url, fields, { signal } = {}) {
        const imageNameForRequest = currentImage ? currentImage.name : null;
        const variantForRequest = samVariant;
        const preloadToken = await waitForSamPreloadIfActive(imageNameForRequest, variantForRequest);
        let payload = await buildSamImagePayload({ variantOverride: variantForRequest, preferredToken: preloadToken });
        let resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...fields, ...payload }),
            signal,
        });
        if (resp.status === 428) {
            payload = await buildSamImagePayload({ forceBase64: true, variantOverride: variantForRequest, preferredToken: preloadToken });
            resp = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ...fields, ...payload }),
                signal,
            });
        }
        return resp;
    }

    /*****************************************************
     * Existing SAM / CLIP calls
     *****************************************************/
    async function sam2BboxPrompt(bbox) {
        const statusToken = beginSamActionStatus("Running SAM box prompt…");
        const imageName = currentImage ? currentImage.name : null;
        const placeholderContext = bbox ? { uuid: bbox.uuid, imageName } : null;
        const jobHandle = registerSamJob({
            type: "sam-bbox",
            imageName,
            cleanup: () => {
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
            },
        });
        try {
            const bodyFields = {
                bbox_left: bbox.x,
                bbox_top: bbox.y,
                bbox_width: bbox.width,
                bbox_height: bbox.height,
                uuid: bbox.uuid,
                sam_variant: samVariant,
            };
            let resp = await postSamEndpoint(`${API_ROOT}/sam2_bbox`, bodyFields);
            if (!resp.ok) {
                throw new Error("Response not OK: " + resp.statusText);
            }
            const result = await resp.json();
            if (!isSamJobActive(jobHandle)) {
                return;
            }
            if (result.image_token && currentImage) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            const returnedUUID = result.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null
            };

            if (result.bbox) {
                const [cx, cy, ww, hh] = result.bbox;
                const absW = ww * currentImage.width;
                const absH = hh * currentImage.height;
                const absX = cx * currentImage.width - absW / 2;
                const absY = cy * currentImage.height - absH / 2;
                targetBbox.x = absX;
                targetBbox.y = absY;
                targetBbox.width = absW;
                targetBbox.height = absH;
                updateBboxAfterTransform();
                console.log("Updated SAM bounding box:", absX, absY, absW, absH);
            } else {
                console.warn("No 'bbox' field returned from sam2_bbox. Full response:", result);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            delete pendingApiBboxes[returnedUUID];
        } catch (err) {
            console.error("sam2_bbox error:", err);
            alert("sam2_bbox call failed: " + err);
            if (placeholderContext) {
                removePendingBbox(placeholderContext);
            }
        } finally {
            completeSamJob(jobHandle.id);
            endSamActionStatus(statusToken);
        }
    }

    async function sam2PointPrompt(px, py) {
        const statusToken = beginSamActionStatus("Running SAM point prompt…");
        const imageName = currentImage ? currentImage.name : null;
        const placeholderContext = currentBbox && currentBbox.bbox ? { uuid: currentBbox.bbox.uuid, imageName } : null;
        const jobHandle = registerSamJob({
            type: "sam-point",
            imageName,
            cleanup: () => {
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
            },
        });
        try {
            const bodyFields = {
                point_x: px,
                point_y: py,
                uuid: currentBbox ? currentBbox.bbox.uuid : null,
                sam_variant: samVariant,
            };
            let resp = await postSamEndpoint(`${API_ROOT}/sam2_point`, bodyFields);
            if (!resp.ok) {
                throw new Error("sam2_point failed: " + resp.statusText);
            }
            const result = await resp.json();
            if (!isSamJobActive(jobHandle)) {
                return;
            }
            if (result.image_token && currentImage) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            const returnedUUID = result.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null
            };

            if (!result.bbox) {
                console.warn("No 'bbox' field in sam2_point response:", result);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            const [cx, cy, wNorm, hNorm] = result.bbox;
            const absW = wNorm * currentImage.width;
            const absH = hNorm * currentImage.height;
            const absX = cx * currentImage.width - absW / 2;
            const absY = cy * currentImage.height - absH / 2;
            targetBbox.x = absX;
            targetBbox.y = absY;
            targetBbox.width = absW;
            targetBbox.height = absH;
            updateBboxAfterTransform();
            console.log("Updated existing bbox from point mode:", absX, absY, absW, absH);
            delete pendingApiBboxes[returnedUUID];
        } catch (err) {
            console.error("sam2PointPrompt error:", err);
            alert("sam2PointPrompt call failed: " + err);
            if (placeholderContext) {
                removePendingBbox(placeholderContext);
            }
        } finally {
            completeSamJob(jobHandle.id);
            endSamActionStatus(statusToken);
        }
    }

    async function autoPredictNewCrop(bbox) {
        const progressToken = beginClipProgress();
        try {
            const offCanvas = document.createElement("canvas");
            offCanvas.width = bbox.width;
            offCanvas.height = bbox.height;
            const ctx = offCanvas.getContext("2d");
            ctx.drawImage(
                currentImage.object,
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height,
                0,
                0,
                bbox.width,
                bbox.height
            );
            const base64String = offCanvas.toDataURL("image/jpeg");
            const base64Data = base64String.split(",")[1];
            const resp = await fetch("http://localhost:8000/predict_base64", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image_base64: base64Data,
                    uuid: bbox.uuid
                })
            });
            const data = await resp.json();
            console.log("autoPredictNewCrop =>", data);
            if (!data.uuid) {
                alert("Auto mode error: you probably don't have a trained .pkl file for CLIP!");
                removeBbox(bbox);
                return;
            }
            const predictedClass = data.prediction;
            const returnedUUID = data.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null
            };
            const oldClass = targetBbox.class;
            const oldArr = bboxes[currentImage.name][oldClass];
            const idx = oldArr.indexOf(targetBbox);
            if (idx !== -1) oldArr.splice(idx, 1);
            if (typeof classes[predictedClass] === "undefined") {
                console.warn("AutoPredict returned unknown class:", predictedClass);
                if (!bboxes[currentImage.name][oldClass]) {
                    bboxes[currentImage.name][oldClass] = [];
                }
                targetBbox.class = oldClass;
                bboxes[currentImage.name][oldClass].push(targetBbox);
            } else {
                if (!bboxes[currentImage.name][predictedClass]) {
                    bboxes[currentImage.name][predictedClass] = [];
                }
                targetBbox.class = predictedClass;
                bboxes[currentImage.name][predictedClass].push(targetBbox);
            }
            delete pendingApiBboxes[returnedUUID];
        } finally {
            endClipProgress(progressToken);
        }
    }

    async function sam2BboxAutoPrompt(bbox) {
        const statusToken = beginSamActionStatus("Running SAM auto box…");
        const imageName = currentImage ? currentImage.name : null;
        const placeholderContext = bbox ? { uuid: bbox.uuid, imageName } : null;
        const jobHandle = registerSamJob({
            type: "sam-bbox-auto",
            imageName,
            cleanup: () => {
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
            },
        });
        try {
            const useFallback = document.getElementById("useFallbackDilate")?.checked;
            const minProbaEl = document.getElementById("minProba");
            const dilateRatioEl = document.getElementById("dilateRatio");
            const bodyData = {
                bbox_left: bbox.x,
                bbox_top: bbox.y,
                bbox_width: bbox.width,
                bbox_height: bbox.height,
                uuid: bbox.uuid,
                sam_variant: samVariant,
            };
            if (useFallback) {
                bodyData.clip_crop_policy = "dilate_on_low_conf";
                if (minProbaEl && !isNaN(parseFloat(minProbaEl.value))) bodyData.fallback_min_proba = parseFloat(minProbaEl.value);
                if (dilateRatioEl && !isNaN(parseFloat(dilateRatioEl.value))) bodyData.fallback_dilate_ratio = parseFloat(dilateRatioEl.value);
            }
            let resp = await postSamEndpoint(`${API_ROOT}/sam2_bbox_auto`, bodyData);
            if (!resp.ok) {
                throw new Error("sam2_bbox_auto failed: " + resp.statusText);
            }
            const result = await resp.json();
            console.log("sam2_bbox_auto =>", result);
            if (!isSamJobActive(jobHandle)) {
                return;
            }
            if (result.image_token && currentImage) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            if (!result.uuid || !result.bbox || result.bbox.length < 4) {
                alert("Auto mode error: missing 'uuid' or invalid 'bbox' in /sam2_bbox_auto response.");
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            const returnedUUID = result.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null
            };
        const [cx, cy, wNorm, hNorm] = result.bbox;
        const absW = wNorm * currentImage.width;
        const absH = hNorm * currentImage.height;
        const absX = cx * currentImage.width - absW / 2;
        const absY = cy * currentImage.height - absH / 2;
        const oldClass = targetBbox.class;
        const oldArr = bboxes[currentImage.name][oldClass];
        const idx = oldArr.indexOf(targetBbox);
        if (idx !== -1) oldArr.splice(idx, 1);
        // If prediction present, move bbox to predicted class; else keep current class
        if (result.prediction) {
            const newClass = result.prediction;
            if (typeof classes[newClass] === "undefined") {
                console.warn("SAM bbox auto predicted unknown class:", newClass);
                if (!bboxes[currentImage.name][oldClass]) {
                    bboxes[currentImage.name][oldClass] = [];
                }
                targetBbox.class = oldClass;
                bboxes[currentImage.name][oldClass].push(targetBbox);
            } else {
                if (!bboxes[currentImage.name][newClass]) {
                    bboxes[currentImage.name][newClass] = [];
                }
                targetBbox.class = newClass;
                bboxes[currentImage.name][newClass].push(targetBbox);
            }
        } else {
            // Keep original class
            if (!bboxes[currentImage.name][oldClass]) {
                bboxes[currentImage.name][oldClass] = [];
            }
            bboxes[currentImage.name][oldClass].push(targetBbox);
        }
        targetBbox.x = absX;
        targetBbox.y = absY;
        targetBbox.width = absW;
        targetBbox.height = absH;
        updateBboxAfterTransform();
        delete pendingApiBboxes[returnedUUID];
        } catch (err) {
            console.error("sam2_bbox_auto error:", err);
            alert("sam2_bbox_auto call failed: " + err);
            if (placeholderContext) {
                removePendingBbox(placeholderContext);
            }
        } finally {
            completeSamJob(jobHandle.id);
            endSamActionStatus(statusToken);
        }
    }

    async function sam2PointAutoPrompt(px, py) {
        const statusToken = beginSamActionStatus("Running SAM point+CLIP…");
        const imageName = currentImage ? currentImage.name : null;
        const placeholderContext = currentBbox && currentBbox.bbox ? { uuid: currentBbox.bbox.uuid, imageName } : null;
        const jobHandle = registerSamJob({
            type: "sam-point-auto",
            imageName,
            cleanup: () => {
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
            },
        });
        try {
            const useFallback = document.getElementById("useFallbackDilate")?.checked;
            const minProbaEl = document.getElementById("minProba");
            const dilateRatioEl = document.getElementById("dilateRatio");
            const bodyData = {
                point_x: px,
                point_y: py,
                uuid: currentBbox ? currentBbox.bbox.uuid : null,
                sam_variant: samVariant,
            };
            if (useFallback) {
                bodyData.clip_crop_policy = "dilate_on_low_conf";
                if (minProbaEl && !isNaN(parseFloat(minProbaEl.value))) bodyData.fallback_min_proba = parseFloat(minProbaEl.value);
                if (dilateRatioEl && !isNaN(parseFloat(dilateRatioEl.value))) bodyData.fallback_dilate_ratio = parseFloat(dilateRatioEl.value);
            }
            let resp = await postSamEndpoint(`${API_ROOT}/sam2_point_auto`, bodyData);
            if (!resp.ok) {
                throw new Error("sam2_point_auto failed: " + resp.statusText);
            }
            const result = await resp.json();
            console.log("sam2_point_auto =>", result);
            if (!isSamJobActive(jobHandle)) {
                return;
            }
            if (result.image_token && currentImage) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            if (!result.uuid || !result.bbox || result.bbox.length < 4) {
                alert("Auto mode error: missing 'uuid' or invalid 'bbox' in /sam2_point_auto response.");
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            const returnedUUID = result.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null
            };
        const [cx, cy, wNorm, hNorm] = result.bbox;
        const absW = wNorm * currentImage.width;
        const absH = hNorm * currentImage.height;
        const absX = cx * currentImage.width - absW / 2;
        const absY = cy * currentImage.height - absH / 2;
        const oldClass = targetBbox.class;
        const oldArr = bboxes[currentImage.name][oldClass];
        const idx = oldArr.indexOf(targetBbox);
        if (idx !== -1) oldArr.splice(idx, 1);
        if (result.prediction) {
            const newClass = result.prediction;
            if (typeof classes[newClass] === "undefined") {
                console.warn("SAM point auto predicted unknown class:", newClass);
                if (!bboxes[currentImage.name][oldClass]) {
                    bboxes[currentImage.name][oldClass] = [];
                }
                targetBbox.class = oldClass;
                bboxes[currentImage.name][oldClass].push(targetBbox);
            } else {
                if (!bboxes[currentImage.name][newClass]) {
                    bboxes[currentImage.name][newClass] = [];
                }
                targetBbox.class = newClass;
                bboxes[currentImage.name][newClass].push(targetBbox);
            }
        } else {
            if (!bboxes[currentImage.name][oldClass]) {
                bboxes[currentImage.name][oldClass] = [];
            }
            bboxes[currentImage.name][oldClass].push(targetBbox);
        }
        targetBbox.x = absX;
        targetBbox.y = absY;
        targetBbox.width = absW;
        targetBbox.height = absH;
        updateBboxAfterTransform();
        delete pendingApiBboxes[returnedUUID];
        } catch (err) {
            console.error("sam2_point_auto error:", err);
            alert("sam2_point_auto call failed: " + err);
            if (placeholderContext) {
                removePendingBbox(placeholderContext);
            }
        } finally {
            completeSamJob(jobHandle.id);
            endSamActionStatus(statusToken);
        }
    }

    async function sam2PointMultiPrompt(job, jobHandle) {
        const { positivePoints, negativePoints, requestToken, placeholderContext } = job;
        const statusToken = beginSamActionStatus("Running SAM multi-point…");
        let success = false;
        try {
            const bodyData = {
                positive_points: positivePoints,
                negative_points: negativePoints,
                uuid: placeholderContext ? placeholderContext.uuid : null,
                sam_variant: samVariant,
            };
            const resp = await postSamEndpoint(`${API_ROOT}/sam2_point_multi`, bodyData);
            if (!resp.ok) {
                throw new Error("sam2_point_multi failed: " + resp.statusText);
            }
            const result = await resp.json();
            if (!isSamJobActive(jobHandle)) {
                return;
            }
            if (requestToken && multiPointPendingToken !== requestToken) {
                return;
            }
            if (result.image_token && currentImage) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            const returnedUUID = result.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                removePendingBbox(placeholderContext);
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null,
            };
            if (!result.bbox || result.bbox.length < 4) {
                console.warn("No 'bbox' field in sam2_point_multi response:", result);
                removePendingBbox(placeholderContext);
                return;
            }
            const [cx, cy, wNorm, hNorm] = result.bbox;
            const absW = wNorm * currentImage.width;
            const absH = hNorm * currentImage.height;
            const absX = cx * currentImage.width - absW / 2;
            const absY = cy * currentImage.height - absH / 2;
            targetBbox.x = absX;
            targetBbox.y = absY;
            targetBbox.width = absW;
            targetBbox.height = absH;
            updateBboxAfterTransform();
            delete pendingApiBboxes[returnedUUID];
            success = true;
            if (multiPointPendingBboxInfo && returnedUUID === multiPointPendingBboxInfo.uuid) {
                multiPointPendingBboxInfo = null;
            }
        } catch (err) {
            console.error("sam2_point_multi error:", err);
            alert("sam2_point_multi call failed: " + err);
            removePendingBbox(placeholderContext);
        } finally {
            endSamActionStatus(statusToken);
            if (!success && placeholderContext) {
                if (multiPointPendingBboxInfo && multiPointPendingBboxInfo.uuid === placeholderContext.uuid && multiPointPendingBboxInfo.imageName === placeholderContext.imageName) {
                    multiPointPendingBboxInfo = null;
                }
            }
        }
    }

    async function sam2PointMultiAutoPrompt(job, jobHandle) {
        const { positivePoints, negativePoints, requestToken, placeholderContext } = job;
        const statusToken = beginSamActionStatus("Running SAM multi-point auto…");
        let success = false;
        try {
            const useFallback = document.getElementById("useFallbackDilate")?.checked;
            const minProbaEl = document.getElementById("minProba");
            const dilateRatioEl = document.getElementById("dilateRatio");
            const bodyData = {
                positive_points: positivePoints,
                negative_points: negativePoints,
                uuid: placeholderContext ? placeholderContext.uuid : null,
                sam_variant: samVariant,
            };
            if (useFallback) {
                bodyData.clip_crop_policy = "dilate_on_low_conf";
                if (minProbaEl && !isNaN(parseFloat(minProbaEl.value))) {
                    bodyData.fallback_min_proba = parseFloat(minProbaEl.value);
                }
                if (dilateRatioEl && !isNaN(parseFloat(dilateRatioEl.value))) {
                    bodyData.fallback_dilate_ratio = parseFloat(dilateRatioEl.value);
                }
            }
            const resp = await postSamEndpoint(`${API_ROOT}/sam2_point_multi_auto`, bodyData);
            if (!resp.ok) {
                throw new Error("sam2_point_multi_auto failed: " + resp.statusText);
            }
            const result = await resp.json();
            console.log("sam2_point_multi_auto =>", result);
            if (!isSamJobActive(jobHandle)) {
                return;
            }
            if (requestToken && multiPointPendingToken !== requestToken) {
                return;
            }
            if (result.image_token && currentImage) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            if (!result.uuid || !result.bbox || result.bbox.length < 4) {
                alert("Auto mode error: missing 'uuid' or invalid 'bbox' in /sam2_point_multi_auto response.");
                removePendingBbox(placeholderContext);
                return;
            }
            const returnedUUID = result.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                removePendingBbox(placeholderContext);
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null,
            };
            const [cx, cy, wNorm, hNorm] = result.bbox;
            const absW = wNorm * currentImage.width;
            const absH = hNorm * currentImage.height;
            const absX = cx * currentImage.width - absW / 2;
            const absY = cy * currentImage.height - absH / 2;
            const oldClass = targetBbox.class;
            const oldArr = bboxes[currentImage.name][oldClass];
            const idx = oldArr.indexOf(targetBbox);
            if (idx !== -1) oldArr.splice(idx, 1);
            if (result.prediction) {
                const newClass = result.prediction;
                if (typeof classes[newClass] === "undefined") {
                    console.warn("SAM multi-point auto predicted unknown class:", newClass);
                    if (!bboxes[currentImage.name][oldClass]) {
                        bboxes[currentImage.name][oldClass] = [];
                    }
                    targetBbox.class = oldClass;
                    bboxes[currentImage.name][oldClass].push(targetBbox);
                } else {
                    if (!bboxes[currentImage.name][newClass]) {
                        bboxes[currentImage.name][newClass] = [];
                    }
                    targetBbox.class = newClass;
                    bboxes[currentImage.name][newClass].push(targetBbox);
                }
            } else {
                if (!bboxes[currentImage.name][oldClass]) {
                    bboxes[currentImage.name][oldClass] = [];
                }
                bboxes[currentImage.name][oldClass].push(targetBbox);
            }
            targetBbox.x = absX;
            targetBbox.y = absY;
            targetBbox.width = absW;
            targetBbox.height = absH;
            updateBboxAfterTransform();
            delete pendingApiBboxes[returnedUUID];
            success = true;
            if (multiPointPendingBboxInfo && returnedUUID === multiPointPendingBboxInfo.uuid) {
                multiPointPendingBboxInfo = null;
            }
        } catch (err) {
            console.error("sam2_point_multi_auto error:", err);
            alert("sam2_point_multi_auto call failed: " + err);
            removePendingBbox(placeholderContext);
        } finally {
            endSamActionStatus(statusToken);
            if (!success && placeholderContext) {
                if (multiPointPendingBboxInfo && multiPointPendingBboxInfo.uuid === placeholderContext.uuid && multiPointPendingBboxInfo.imageName === placeholderContext.imageName) {
                    multiPointPendingBboxInfo = null;
                }
            }
        }
    }

    function addMultiPointAnnotation(label) {
        if (!multiPointMode || !currentImage || currentClass === null) {
            return;
        }
        const clampedX = Math.max(0, Math.min(currentImage.width, mouse.realX));
        const clampedY = Math.max(0, Math.min(currentImage.height, mouse.realY));
        multiPointPoints.push({ x: clampedX, y: clampedY, label });
    }

    function drawMultiPointMarkers(context) {
        if (multiPointPoints.length === 0) {
            return;
        }
        const prevLineWidth = context.lineWidth;
        const prevFont = context.font;
        const prevAlign = context.textAlign;
        const prevBaseline = context.textBaseline;
        multiPointPoints.forEach((pt) => {
            const palette = pt.label === 1 ? multiPointColors.positive : multiPointColors.negative;
            const radius = Math.max(3.5, 5 * scale);
            context.beginPath();
            context.strokeStyle = palette.stroke;
            context.fillStyle = palette.fill;
            context.lineWidth = Math.max(1.2, 1.2 * scale);
            context.arc(zoomX(pt.x), zoomY(pt.y), radius, 0, Math.PI * 2);
            context.fill();
            context.stroke();
            context.fillStyle = palette.stroke;
            context.font = context.font.replace(/\d+px/, `${Math.max(9, zoom(12))}px`);
            context.textAlign = "center";
            context.textBaseline = "middle";
            context.fillText(pt.label === 1 ? "+" : "-", zoomX(pt.x), zoomY(pt.y));
        });
        context.lineWidth = prevLineWidth;
        context.font = prevFont;
        context.textAlign = prevAlign;
        context.textBaseline = prevBaseline;
    }

    async function submitMultiPointSelection() {
        if (!multiPointMode) {
            return;
        }
        if (!currentImage || currentClass === null) {
            alert("Select an image and class before using multi-point mode.");
            return;
        }
        const positivePoints = multiPointPoints.filter((pt) => pt.label === 1);
        if (positivePoints.length === 0) {
            alert("Add at least one positive point before submitting.");
            return;
        }
        const negativePoints = multiPointPoints.filter((pt) => pt.label === 0);
        const positivePayload = positivePoints.map((pt) => [pt.x, pt.y]);
        const negativePayload = negativePoints.map((pt) => [pt.x, pt.y]);
        const requestToken = generateUUID();
        const primary = positivePoints[0];
        const dotSize = 10;
        const half = dotSize / 2;
        const originalMouse = {
            startRealX: mouse.startRealX,
            startRealY: mouse.startRealY,
            realX: mouse.realX,
            realY: mouse.realY,
        };
        mouse.startRealX = primary.x - half;
        mouse.startRealY = primary.y - half;
        mouse.realX = primary.x + half;
        mouse.realY = primary.y + half;
        storeNewBbox(dotSize, dotSize);
        updateBboxAfterTransform();
        const placeholderContext = currentBbox && currentBbox.bbox && currentImage
            ? { uuid: currentBbox.bbox.uuid, imageName: currentImage.name }
            : null;
        mouse.startRealX = originalMouse.startRealX;
        mouse.startRealY = originalMouse.startRealY;
        mouse.realX = originalMouse.realX;
        mouse.realY = originalMouse.realY;
        multiPointPoints = [];
        enqueueMultiPointJob({
            requestToken,
            positivePoints: positivePayload,
            negativePoints: negativePayload,
            placeholderContext,
            imageName: currentImage.name,
            auto: samMultiPointAutoMode,
        });
    }

    // Standard parameters
    const fontBaseSize = 6;
    const fontColor = "#001f3f";
    const borderColor = "#001f3f";
    const backgroundColor = "rgba(0, 116, 217, 0.2)";
    const markedFontColor = "#FF4136";
    const markedBorderColor = "#FF4136";
    const markedBackgroundColor = "rgba(255, 133, 27, 0.2)";
    const minBBoxWidth = 5;
    const minBBoxHeight = 5;
    const scrollSpeed = 1.03;
    const minZoom = 0.1;
    const maxZoom = 5;
    const edgeSize = 5;
    const resetCanvasOnChange = true;
    const defaultScale = 0.5;
    const drawCenterX = true;
    const drawGuidelines = true;
    const fittedZoom = true;
    let canvas = null;
    let images = {};
    let classes = {};
    let bboxes = {};
    const extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"];
    let currentImage = null;
    let currentClass = null;
    let currentBbox = null;
    let imageListIndex = 0;
    let classListIndex = 0;
    let scale = defaultScale;
    let canvasX = 0;
    let canvasY = 0;
    let screenX = 0;
    let screenY = 0;
    const mouse = {
        x: 0,
        y: 0,
        realX: 0,
        realY: 0,
        buttonL: false,
        buttonR: false,
        startRealX: 0,
        startRealY: 0
    };

    document.addEventListener("contextmenu", function (e) {
        e.preventDefault();
    }, false);

    document.onreadystatechange = () => {
        if (document.readyState === "complete") {
            listenCanvas();
            listenCanvasMouse();
            listenImageLoad();
            listenImageSelect();
            listenClassLoad();
            listenClassSelect();
            listenBboxLoad();
            listenBboxSave();
            listenKeyboard();
            listenImageSearch();
            listenImageCrop();
        }
    };

    function ensureCanvasDimensions() {
        if (!canvas || !canvas.element) {
            return;
        }
        const parent = canvas.element.parentElement;
        const fallbackWidth = parent ? parent.clientWidth : 0;
        const fallbackHeight = parent ? parent.clientHeight : 0;
        const measuredWidth = canvas.element.clientWidth || fallbackWidth || document.getElementById("right")?.clientWidth || window.innerWidth;
        const measuredHeight = canvas.element.clientHeight || fallbackHeight || window.innerHeight - 20;
        const targetWidth = Math.max(1, measuredWidth);
        const targetHeight = Math.max(1, measuredHeight);
        if (canvas.width !== targetWidth) {
            canvas.width = targetWidth;
            canvas.element.width = targetWidth;
        }
        if (canvas.height !== targetHeight) {
            canvas.height = targetHeight;
            canvas.element.height = targetHeight;
        }
    }

    const listenCanvas = () => {
        const rightPanel = document.getElementById("right");
        const initialWidth = rightPanel ? rightPanel.clientWidth : window.innerWidth;
        const initialHeight = Math.max(1, window.innerHeight - 20);
        canvas = new Canvas("canvas", initialWidth, initialHeight);
        ensureCanvasDimensions();
        canvas.on("draw", (context) => {
            if (currentImage !== null) {
                drawImage(context);
                drawNewBbox(context);
                drawExistingBboxes(context);
                drawMultiPointMarkers(context);
                drawCross(context);
            } else {
                drawIntro(context);
            }
        }).start();
        window.addEventListener("resize", () => {
            ensureCanvasDimensions();
            if (currentImage) {
                fitZoom(currentImage, { preservePan: true });
            }
        });
    };

    const drawImage = (context) => {
        context.drawImage(
            currentImage.object,
            zoomX(0),
            zoomY(0),
            zoom(currentImage.width),
            zoom(currentImage.height)
        );
    };

    const drawIntro = (context) => {
        setFontStyles(context, false);
        context.fillText("USAGE:", zoomX(20), zoomY(50));
        context.fillText("1. Load your images (jpg, png).", zoomX(20), zoomY(100));
        context.fillText("2. Load your classes (yolo *.names).", zoomX(20), zoomY(150));
        context.fillText("3. Create bboxes or restore from zipped YOLO annotations.", zoomX(20), zoomY(200));
        context.fillText("NOTES:", zoomX(20), zoomY(300));
        context.fillText("1: Reloading images resets bboxes.", zoomX(20), zoomY(350));
        context.fillText("2: Check out README.md for more info.", zoomX(20), zoomY(400));
    };

    const drawNewBbox = (context) => {
        if (mouse.buttonL === true && currentClass !== null && currentBbox === null) {
            const width = (mouse.realX - mouse.startRealX);
            const height = (mouse.realY - mouse.startRealY);
            const strokeColor = getColorFromClass(currentClass);
            const fillColor = withAlpha(strokeColor, 0.2);
            context.setLineDash([]);
            context.strokeStyle = strokeColor;
            context.fillStyle = fillColor;
            context.strokeRect(
                zoomX(mouse.startRealX),
                zoomY(mouse.startRealY),
                zoom(width),
                zoom(height)
            );
            context.fillRect(
                zoomX(mouse.startRealX),
                zoomY(mouse.startRealY),
                zoom(width),
                zoom(height)
            );
            drawX(context, mouse.startRealX, mouse.startRealY, width, height);
            setBboxCoordinates(mouse.startRealX, mouse.startRealY, width, height);
        }
    };

    const drawExistingBboxes = (context) => {
        const currentBboxes = bboxes[currentImage.name];
        if (!currentBboxes) {
            return;
        }
        for (let className in currentBboxes) {
            currentBboxes[className].forEach((bbox) => {
                context.save();
                const strokeColor = getColorFromClass(className);
                const fillColor = withAlpha(strokeColor, 0.2);
                const isCurrent = currentBbox && currentBbox.bbox === bbox;
                const lineWidth = isCurrent ? Math.max(1.5, 1.5 * scale) : 1;

                context.font = context.font.replace(/\d+px/, `${Math.max(8, zoom(fontBaseSize))}px`);
                context.fillStyle = strokeColor;
                context.fillText(className, zoomX(bbox.x), zoomY(bbox.y - 2));

                context.setLineDash([]);
                context.lineWidth = lineWidth;
                context.strokeStyle = strokeColor;
                context.fillStyle = fillColor;
                context.fillRect(
                    zoomX(bbox.x),
                    zoomY(bbox.y),
                    zoom(bbox.width),
                    zoom(bbox.height)
                );
                context.strokeRect(
                    zoomX(bbox.x),
                    zoomY(bbox.y),
                    zoom(bbox.width),
                    zoom(bbox.height)
                );
                drawX(context, bbox.x, bbox.y, bbox.width, bbox.height);

                if (isCurrent && currentBbox.resizing) {
                    const handlePoint = getCornerCoordinates(bbox, currentBbox.resizing);
                    if (handlePoint) {
                        drawCornerHandle(context, handlePoint.x, handlePoint.y, strokeColor);
                    }
                }

                if (bbox.marked === true) {
                    setBboxCoordinates(bbox.x, bbox.y, bbox.width, bbox.height);
                }
                context.restore();
            });
        }
    };

    function getCornerCoordinates(bbox, corner) {
        const x1 = bbox.x;
        const y1 = bbox.y;
        const x2 = bbox.x + bbox.width;
        const y2 = bbox.y + bbox.height;
        switch (corner) {
            case "topLeft":
                return { x: x1, y: y1 };
            case "topRight":
                return { x: x2, y: y1 };
            case "bottomLeft":
                return { x: x1, y: y2 };
            case "bottomRight":
                return { x: x2, y: y2 };
            default:
                return null;
        }
    }

    function drawCornerHandle(context, x, y, strokeColor) {
        context.save();
        const radius = Math.max(3, 5 * scale);
        context.beginPath();
        context.strokeStyle = strokeColor;
        context.fillStyle = withAlpha(strokeColor, 0.35);
        context.lineWidth = Math.max(1.2, 1.2 * scale);
        context.arc(zoomX(x), zoomY(y), radius, 0, Math.PI * 2);
        context.fill();
        context.stroke();
        context.restore();
    }

    const drawX = (context, x, y, width, height) => {
        if (drawCenterX === true) {
            const centerX = x + width / 2;
            const centerY = y + height / 2;
            context.beginPath();
            context.moveTo(zoomX(centerX), zoomY(centerY - 10));
            context.lineTo(zoomX(centerX), zoomY(centerY + 10));
            context.stroke();
            context.beginPath();
            context.moveTo(zoomX(centerX - 10), zoomY(centerY));
            context.lineTo(zoomX(centerX + 10), zoomY(centerY));
            context.stroke();
        }
    };

    const drawCross = (context) => {
        if (drawGuidelines === true) {
            context.setLineDash([5]);
            context.beginPath();
            context.moveTo(zoomX(mouse.realX), zoomY(0));
            context.lineTo(zoomX(mouse.realX), zoomY(currentImage.height));
            context.stroke();
            context.beginPath();
            context.moveTo(zoomX(0), zoomY(mouse.realY));
            context.lineTo(zoomX(currentImage.width), zoomY(mouse.realY));
            context.stroke();
        }
    };

    const setBBoxStyles = (context, marked) => {
        context.setLineDash([]);
        if (marked === false) {
            context.strokeStyle = borderColor;
            context.fillStyle = backgroundColor;
        } else {
            context.strokeStyle = markedBorderColor;
            context.fillStyle = markedBackgroundColor;
        }
    };

    const setBboxCoordinates = (x, y, width, height) => {
        const x2 = x + width;
        const y2 = y + height;
        document.getElementById("bboxInformation").innerHTML =
            `${width}x${height} (${x}, ${y}) (${x2}, ${y2})`;
    };

    const setFontStyles = (context, marked) => {
        if (marked === false) {
            context.fillStyle = fontColor;
        } else {
            context.fillStyle = markedFontColor;
        }
        context.font = context.font.replace(/\d+px/, `${zoom(fontBaseSize)}px`);
    };

    const listenCanvasMouse = () => {
        canvas.element.addEventListener("wheel", trackWheel, { passive: false });
        canvas.element.addEventListener("mousemove", trackPointer);
        canvas.element.addEventListener("mousedown", trackPointer);
        canvas.element.addEventListener("mouseup", trackPointer);
        canvas.element.addEventListener("mouseout", trackPointer);
    };

    const trackWheel = (event) => {
        if (event.shiftKey) {
            const panSpeed = -1.5;
            canvasX -= event.deltaX * panSpeed;
            canvasY -= event.deltaY * panSpeed;
            mouse.realX = zoomXInv(mouse.x);
            mouse.realY = zoomYInv(mouse.y);
            event.preventDefault();
            return;
        }
        if (event.deltaY < 0) {
            scale = Math.min(maxZoom, scale * scrollSpeed);
        } else {
            scale = Math.max(minZoom, scale * (1 / scrollSpeed));
        }
        canvasX = mouse.realX;
        canvasY = mouse.realY;
        screenX = mouse.x;
        screenY = mouse.y;
        mouse.realX = zoomXInv(mouse.x);
        mouse.realY = zoomYInv(mouse.y);
        event.preventDefault();
    };

    async function trackPointer(event) {
        mouse.bounds = canvas.element.getBoundingClientRect();
        mouse.x = event.clientX - mouse.bounds.left;
        mouse.y = event.clientY - mouse.bounds.top;
        const oldRealX = mouse.realX;
        const oldRealY = mouse.realY;
        mouse.realX = zoomXInv(mouse.x);
        mouse.realY = zoomYInv(mouse.y);
    
        if (event.type === "mousedown") {
            mouse.startRealX = mouse.realX;
            mouse.startRealY = mouse.realY;
            if (event.which === 3) {
                mouse.buttonR = true;
            } else if (event.which === 1) {
                mouse.buttonL = true;
            }
        }
        else if (event.type === "mouseup" || event.type === "mouseout") {
            if (mouse.buttonL && currentImage !== null && currentClass !== null) {
                if (multiPointMode) {
                    mouse.buttonL = false;
                    mouse.buttonR = false;
                    return;
                }
                if (pointMode) {
                    currentBbox = null;
                    const dotSize = 10;
                    const half = dotSize / 2;
                    mouse.startRealX = mouse.realX - half;
                    mouse.startRealY = mouse.realY - half;
                    storeNewBbox(dotSize, dotSize);
                    mouse.buttonL = false;
                    mouse.buttonR = false;
                    if (samPointAutoMode) {
                        await sam2PointAutoPrompt(mouse.realX, mouse.realY);
                    } else {
                        await sam2PointPrompt(mouse.realX, mouse.realY);
                    }
                    setBboxMarkedState();
                    if (currentBbox) {
                        updateBboxAfterTransform();
                    }
                }
                else {
                    const movedWidth  = Math.abs(mouse.realX - mouse.startRealX);
                    const movedHeight = Math.abs(mouse.realY - mouse.startRealY);
                    if (movedWidth > minBBoxWidth && movedHeight > minBBoxHeight) {
                        if (currentBbox === null) {
                            storeNewBbox(movedWidth, movedHeight);
                            mouse.buttonL = false;
                            mouse.buttonR = false;
                            if (samMode && autoMode) {
                                await sam2BboxAutoPrompt(currentBbox.bbox);
                            }
                            else if (autoMode) {
                                await autoPredictNewCrop(currentBbox.bbox);
                            }
                            else if (samMode) {
                                await sam2BboxPrompt(currentBbox.bbox);
                            }
                            else {
                                setBboxMarkedState();
                                if (currentBbox) {
                                    updateBboxAfterTransform();
                                }
                            }
                            setBboxMarkedState();
                            if (currentBbox) {
                                updateBboxAfterTransform();
                            }
                        }
                        else {
                            updateBboxAfterTransform();
                        }
                    }
                    else {
                        if (currentBbox === null) {
                            setBboxMarkedState();
                            if (currentBbox !== null) {
                                updateBboxAfterTransform();
                            }
                        }
                        else {
                            setBboxMarkedState();
                            if (currentBbox !== null) {
                                updateBboxAfterTransform();
                            }
                        }
                    }
                }
            }
            mouse.buttonR = false;
            mouse.buttonL = false;
        }
    
        moveBbox();
        resizeBbox();
        changeCursorByLocation();
        panImage(oldRealX, oldRealY);
    }

    const storeNewBbox = (movedWidth, movedHeight) => {
        const bbox = {
            x: Math.min(mouse.startRealX, mouse.realX),
            y: Math.min(mouse.startRealY, mouse.realY),
            width: movedWidth,
            height: movedHeight,
            marked: true,
            class: currentClass,
            uuid: generateUUID()
        };
        if (!bboxes[currentImage.name]) {
            bboxes[currentImage.name] = {};
        }
        if (!bboxes[currentImage.name][currentClass]) {
            bboxes[currentImage.name][currentClass] = [];
        }
        bboxes[currentImage.name][currentClass].push(bbox);
        currentBbox = {
            bbox: bbox,
            index: bboxes[currentImage.name][currentClass].length - 1,
            originalX: bbox.x,
            originalY: bbox.y,
            originalWidth: bbox.width,
            originalHeight: bbox.height,
            moving: false,
            resizing: null
        };
        pendingApiBboxes[bbox.uuid] = bbox;
    };

    const updateBboxAfterTransform = () => {
        if (currentBbox && currentBbox.resizing !== null) {
            if (currentBbox.bbox.width < 0) {
                currentBbox.bbox.width = Math.abs(currentBbox.bbox.width);
                currentBbox.bbox.x -= currentBbox.bbox.width;
            }
            if (currentBbox.bbox.height < 0) {
                currentBbox.bbox.height = Math.abs(currentBbox.bbox.height);
                currentBbox.bbox.y -= currentBbox.bbox.height;
            }
            currentBbox.resizing = null;
        }
        if (currentBbox) {
            currentBbox.bbox.marked = true;
            currentBbox.originalX = currentBbox.bbox.x;
            currentBbox.originalY = currentBbox.bbox.y;
            currentBbox.originalWidth = currentBbox.bbox.width;
            currentBbox.originalHeight = currentBbox.bbox.height;
            currentBbox.moving = false;
        }
    };

    const setBboxMarkedState = () => {
        if (!currentBbox || (!currentBbox.moving && !currentBbox.resizing)) {
            const currentBxs = bboxes[currentImage.name];
            let wasInside = false;
            let smallestBbox = Number.MAX_SAFE_INTEGER;
            for (let className in currentBxs) {
                currentBxs[className].forEach((bx, i) => {
                    bx.marked = false;
                    const endX = bx.x + bx.width;
                    const endY = bx.y + bx.height;
                    const size = bx.width * bx.height;
                    if (
                        mouse.startRealX >= bx.x && mouse.startRealX <= endX &&
                        mouse.startRealY >= bx.y && mouse.startRealY <= endY
                    ) {
                        wasInside = true;
                        if (size < smallestBbox) {
                            smallestBbox = size;
                            currentBbox = {
                                bbox: bx,
                                index: i,
                                originalX: bx.x,
                                originalY: bx.y,
                                originalWidth: bx.width,
                                originalHeight: bx.height,
                                moving: false,
                                resizing: null
                            };
                        }
                    }
                });
            }
            if (!wasInside) {
                currentBbox = null;
            }
        }
    };

    const moveBbox = () => {
        if (mouse.buttonL && currentBbox) {
            const bx = currentBbox.bbox;
            const endX = bx.x + bx.width;
            const endY = bx.y + bx.height;
            if (
                mouse.startRealX >= bx.x + edgeSize && mouse.startRealX <= endX - edgeSize &&
                mouse.startRealY >= bx.y + edgeSize && mouse.startRealY <= endY - edgeSize
            ) {
                currentBbox.moving = true;
            }
            if (currentBbox.moving) {
                bx.x = currentBbox.originalX + (mouse.realX - mouse.startRealX);
                bx.y = currentBbox.originalY + (mouse.realY - mouse.startRealY);
            }
        }
    };

    const resizeBbox = () => {
        if (mouse.buttonL && currentBbox) {
            const bx = currentBbox.bbox;
            const tlx = bx.x;
            const tly = bx.y;
            const brx = bx.x + bx.width;
            const bry = bx.y + bx.height;

            if (nearCorner(mouse.startRealX, mouse.startRealY, tlx, tly)) {
                currentBbox.resizing = "topLeft";
            } else if (nearCorner(mouse.startRealX, mouse.startRealY, tlx, bry)) {
                currentBbox.resizing = "bottomLeft";
            } else if (nearCorner(mouse.startRealX, mouse.startRealY, brx, tly)) {
                currentBbox.resizing = "topRight";
            } else if (nearCorner(mouse.startRealX, mouse.startRealY, brx, bry)) {
                currentBbox.resizing = "bottomRight";
            }

            if (currentBbox.resizing === "topLeft") {
                bx.x = mouse.realX;
                bx.y = mouse.realY;
                bx.width = currentBbox.originalX + currentBbox.originalWidth - mouse.realX;
                bx.height = currentBbox.originalY + currentBbox.originalHeight - mouse.realY;
            }
            else if (currentBbox.resizing === "bottomLeft") {
                bx.x = mouse.realX;
                bx.y = mouse.realY - (mouse.realY - currentBbox.originalY);
                bx.width = currentBbox.originalX + currentBbox.originalWidth - mouse.realX;
                bx.height = mouse.realY - currentBbox.originalY;
            }
            else if (currentBbox.resizing === "topRight") {
                bx.x = mouse.realX - (mouse.realX - currentBbox.originalX);
                bx.y = mouse.realY;
                bx.width = mouse.realX - currentBbox.originalX;
                bx.height = currentBbox.originalY + currentBbox.originalHeight - mouse.realY;
            }
            else if (currentBbox.resizing === "bottomRight") {
                bx.x = mouse.realX - (mouse.realX - currentBbox.originalX);
                bx.y = mouse.realY - (mouse.realY - currentBbox.originalY);
                bx.width = mouse.realX - currentBbox.originalX;
                bx.height = mouse.realY - currentBbox.originalY;
            }
        }
    };

    function nearCorner(px, py, cx, cy) {
        return (px >= (cx - edgeSize) && px <= (cx + edgeSize) &&
                py >= (cy - edgeSize) && py <= (cy + edgeSize));
    }

    const changeCursorByLocation = () => {
        if (!currentImage) return;
        document.body.style.cursor = "default";
        const currentBxs = bboxes[currentImage.name];
        for (let className in currentBxs) {
            for (let bx of currentBxs[className]) {
                const endX = bx.x + bx.width;
                const endY = bx.y + bx.height;
                if (mouse.realX >= (bx.x + edgeSize) && mouse.realX <= (endX - edgeSize) &&
                    mouse.realY >= (bx.y + edgeSize) && mouse.realY <= (endY - edgeSize)) {
                    document.body.style.cursor = "pointer";
                    break;
                }
            }
        }
        if (currentBbox) {
            const bx = currentBbox.bbox;
            const brx = bx.x + bx.width;
            const bry = bx.y + bx.height;
            if (mouse.realX >= bx.x + edgeSize && mouse.realX <= brx - edgeSize &&
                mouse.realY >= bx.y + edgeSize && mouse.realY <= bry - edgeSize) {
                document.body.style.cursor = "move";
            }
        }
    };

    const panImage = (xx, yy) => {
        if (mouse.buttonR) {
            canvasX -= mouse.realX - xx;
            canvasY -= mouse.realY - yy;
            mouse.realX = zoomXInv(mouse.x);
            mouse.realY = zoomYInv(mouse.y);
        }
    };

    function zoom(n) { return Math.floor(n * scale); }
    function zoomX(n) { return Math.floor((n - canvasX) * scale + screenX); }
    function zoomY(n) { return Math.floor((n - canvasY) * scale + screenY); }
    function zoomXInv(n) { return Math.floor((n - screenX) / scale + canvasX); }
    function zoomYInv(n) { return Math.floor((n - screenY) / scale + canvasY); }

    /******************************************************
     * listenImageLoad
     * We still do one pass for each file to get .width, .height
     * But do NOT store big .object for each file (to save memory).
     ******************************************************/
    const readImageDimensions = async (file) => {
        if (window.createImageBitmap) {
            try {
                const bitmap = await createImageBitmap(file);
                const width = bitmap.width;
                const height = bitmap.height;
                if (typeof bitmap.close === "function") {
                    bitmap.close();
                }
                return { width, height };
            } catch (err) {
                console.warn("createImageBitmap failed", err);
            }
        }
        return new Promise((resolve) => {
            const url = URL.createObjectURL(file);
            const tempImg = new Image();
            tempImg.onload = () => {
                const width = tempImg.width;
                const height = tempImg.height;
                URL.revokeObjectURL(url);
                resolve({ width, height });
            };
            tempImg.onerror = () => {
                URL.revokeObjectURL(url);
                resolve({ width: 0, height: 0 });
            };
            tempImg.src = url;
        });
    };

    const listenImageLoad = () => {
        const imagesInput = document.getElementById("images");
        const bboxesButton = document.getElementById("bboxesSelect");
        if (!imagesInput) {
            return;
        }
        imagesInput.addEventListener("change", (event) => {
            const imageList = document.getElementById("imageList");
            if (!imageList) {
                setSamStatus("Image list element is missing.", { variant: "error", duration: 5000 });
                imagesInput.value = "";
                return;
            }

            const files = event.target.files;
            if (!files || files.length === 0) {
                imagesInput.value = "";
                return;
            }

            resetImageList();
            document.body.style.cursor = "wait";
            let fileCount = 0;
            const promises = [];

            function readDimensions(file) {
                return new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const tempImg = new Image();
                        tempImg.onload = () => {
                            resolve({ width: tempImg.width, height: tempImg.height });
                        };
                        tempImg.src = reader.result;
                    };
                    reader.readAsDataURL(file);
                });
            }

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const nameParts = file.name.split(".");
                const ext = nameParts[nameParts.length - 1];
                if (extensions.indexOf(ext) !== -1) {
                    images[file.name] = {
                        meta: file,
                        index: fileCount,
                        width: 0,
                        height: 0,
                        object: undefined,
                    };
                    fileCount++;
                    const option = document.createElement("option");
                    option.value = file.name;
                    option.innerHTML = file.name;
                    if (fileCount === 1) {
                        option.selected = true;
                    }
                    imageList.appendChild(option);

                    promises.push(
                        readDimensions(file).then((dim) => {
                            images[file.name].width = dim.width;
                            images[file.name].height = dim.height;
                        })
                    );
                }
            }

            if (fileCount === 0) {
                document.body.style.cursor = "default";
                setSamStatus("No supported image files were selected.", { variant: "warn", duration: 5000 });
                imagesInput.value = "";
                return;
            }

            setSamStatus(`Detected ${fileCount} supported image${fileCount === 1 ? "" : "s"}.`, { variant: "success", duration: 2000 });

            Promise.all(promises).then(() => {
                document.body.style.cursor = "default";
                if (fileCount > 0) {
                    const firstName = imageList.options[0].innerHTML;
                    if (!images[firstName]) {
                        setSamStatus(`Failed to stage image data for ${firstName}.`, { variant: "error", duration: 6000 });
                        imagesInput.value = "";
                        return;
                    }
                    setCurrentImage(images[firstName]);
                }
                if (Object.keys(classes).length > 0) {
                    const bboxesInput = document.getElementById("bboxes");
                    if (bboxesInput) {
                        bboxesInput.disabled = false;
                    }
                    setButtonDisabled(bboxesButton, false);
                }
                setSamStatus(`Loaded ${fileCount} image${fileCount === 1 ? "" : "s"}.`, { variant: "success", duration: 3000 });
                imagesInput.value = "";
            });
        });
    };

    const resetImageList = () => {
        document.getElementById("imageList").innerHTML = "";
        images = {};
        bboxes = {};
        currentImage = null;
        samPreloadLastKey = null;
        cancelSamPreload();
        samTokenCache.clear();
        const bboxesInput = document.getElementById("bboxes");
        if (bboxesInput) {
            bboxesInput.disabled = true;
        }
        const bboxesBtn = document.getElementById("bboxesSelect");
        setButtonDisabled(bboxesBtn, true);
        cancelAllSamJobs({ reason: "image reset", announce: false });
        cancelPendingMultiPoint({ clearMarkers: true, removePendingBbox: true });
    };

    function setCurrentImage(image) {
        if (!image) return;
        const previousImageName = currentImage ? currentImage.name : null;
        const cancellation = cancelAllSamJobs({ reason: "image switch", imageName: previousImageName, announce: false });
        cancelPendingMultiPoint({ clearMarkers: true, removePendingBbox: true });
        if (previousImageName) {
            cancelSamPreload();
        }
        const pendingImageName = image?.meta?.name || image?.name || null;
        if (samPreloadEnabled && pendingImageName) {
            samPreloadCurrentImageName = pendingImageName;
            samPreloadCurrentVariant = samVariant;
            setSamStatus(`Preparing SAM preload: ${pendingImageName}`, { variant: "info", duration: 0 });
            showSamPreloadProgress();
        }
        const loadVersion = (image._loadVersion = (image._loadVersion || 0) + 1);
        if (resetCanvasOnChange) {
            resetCanvasPlacement();
        }
        const messagePrefix = cancellation.message;
        if (!image.object) {
            const reader = new FileReader();
            document.body.style.cursor = "wait";
            reader.onload = () => {
                if (image._loadVersion !== loadVersion) {
                    return;
                }
                const dataUrl = typeof reader.result === "string" ? reader.result : "";
                const imageObject = new Image();
                imageObject.onload = () => {
                    if (image._loadVersion !== loadVersion) {
                        return;
                    }
                    image.object = imageObject;
                    image.dataUrl = dataUrl;
                    const naturalWidth = imageObject.naturalWidth || imageObject.width || image.width || 0;
                    const naturalHeight = imageObject.naturalHeight || imageObject.height || image.height || 0;
                    image.width = naturalWidth;
                    image.height = naturalHeight;
                    currentImage = {
                        name: image.meta.name,
                        object: imageObject,
                        width: naturalWidth,
                        height: naturalHeight,
                        dataUrl,
                    };
                    document.body.style.cursor = "default";
                    if (fittedZoom) {
                        fitZoom(currentImage);
                    }
                    document.getElementById("imageInformation").innerHTML =
                        `${naturalWidth}x${naturalHeight}, ${formatBytes(image.meta.size)}`;
                    if (!bboxes[currentImage.name]) {
                        bboxes[currentImage.name] = {};
                    }
                    scheduleSamPreload({ force: true, delayMs: SAM_PRELOAD_IMAGE_SWITCH_DELAY_MS, messagePrefix });
                };
                imageObject.src = dataUrl;
            };
            reader.readAsDataURL(image.meta);
        }
        else {
            if (image._loadVersion !== loadVersion) {
                return;
            }
            const naturalWidth = image.width || image.object?.naturalWidth || image.object?.width || 0;
            const naturalHeight = image.height || image.object?.naturalHeight || image.object?.height || 0;
            image.width = naturalWidth;
            image.height = naturalHeight;
            currentImage = {
                name: image.meta.name,
                object: image.object,
                width: naturalWidth,
                height: naturalHeight,
                dataUrl: image.dataUrl || null,
            };
            if (fittedZoom) {
                fitZoom(currentImage);
            }
            document.getElementById("imageInformation").innerHTML =
                `${naturalWidth}x${naturalHeight}, ${formatBytes(image.meta.size)}`;
            if (!bboxes[currentImage.name]) {
                bboxes[currentImage.name] = {};
            }
            scheduleSamPreload({ force: true, delayMs: SAM_PRELOAD_IMAGE_SWITCH_DELAY_MS, messagePrefix });
        }
        if (currentBbox !== null) {
            currentBbox.bbox.marked = false;
            currentBbox = null;
        }
        if (!samPreloadEnabled && messagePrefix) {
            setSamStatus(messagePrefix, { variant: "warn", duration: 5000 });
        }
    }

    const fitZoom = (image, options = {}) => {
        if (!image) {
            return;
        }
        ensureCanvasDimensions();
        const { preservePan = false } = options;
        const imgWidth = Math.max(1, image.width || image.object?.naturalWidth || image.object?.width || 1);
        const imgHeight = Math.max(1, image.height || image.object?.naturalHeight || image.object?.height || 1);
        const canvasWidth = Math.max(1, canvas?.width || canvas?.element?.width || canvas?.element?.clientWidth || window.innerWidth);
        const canvasHeight = Math.max(1, canvas?.height || canvas?.element?.height || canvas?.element?.clientHeight || (window.innerHeight - 20));
        const nextScale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);
        scale = Math.min(Math.max(nextScale, minZoom), maxZoom);
        if (!preservePan) {
            canvasX = 0;
            canvasY = 0;
            screenX = 0;
            screenY = 0;
        }
    };

    function formatBytes(bytes, decimals) {
        if (bytes === 0) {
            return "0 Bytes";
        }
        const k = 1024;
        const dm = decimals === undefined ? 2 : decimals;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
    }

    const listenImageSelect = () => {
        const imageList = document.getElementById("imageList");
        imageList.addEventListener("change", () => {
            imageListIndex = imageList.selectedIndex;
            const name = imageList.options[imageListIndex].innerHTML;
            setCurrentImage(images[name]);
        });
    };

    const listenClassLoad = () => {
        const classesElement = document.getElementById("classes");
        if (!classesElement) {
            return;
        }
        const classesButton = document.getElementById("classesSelect");
        classesElement.addEventListener("click", () => {
            classesElement.value = null;
        });
        classesElement.addEventListener("change", (event) => {
            const files = event.target.files;
            if (files.length > 0) {
                resetClassList();
                const nameParts = files[0].name.split(".");
                if (nameParts[nameParts.length - 1] === "txt") {
                    const reader = new FileReader();
                    reader.addEventListener("load", () => {
                        const lines = reader.result;
                        const rows = lines.split(/[\r\n]+/);
                        if (rows.length > 0) {
                            const classList = document.getElementById("classList");
                            loadedClassList = [];
                            for (let i = 0; i < rows.length; i++) {
                                rows[i] = rows[i].trim();
                                if (rows[i] !== "") {
                                    classes[rows[i]] = i;
                                    loadedClassList.push(rows[i]);
                                    const option = document.createElement("option");
                                    option.value = i;
                                    option.innerHTML = rows[i];
                                    if (i === 0) {
                                        option.selected = true;
                                        currentClass = rows[i];
                                    }
                                    classList.appendChild(option);
                                }
                            }
                            setCurrentClass();
                            if (Object.keys(images).length > 0) {
                                const bboxesInput = document.getElementById("bboxes");
                                if (bboxesInput) {
                                    bboxesInput.disabled = false;
                                }
                                const bboxesBtn = document.getElementById("bboxesSelect");
                                setButtonDisabled(bboxesBtn, false);
                                document.getElementById("restoreBboxes").disabled = false;
                            }
                        }
                    });
                    reader.readAsText(files[0]);
                }
            }
            classesElement.value = "";
        });
    };

    const resetClassList = () => {
        document.getElementById("classList").innerHTML = "";
        classes = {};
        currentClass = null;
        loadedClassList = [];
        clearMultiPointAnnotations();
    };

    const setCurrentClass = () => {
        const classList = document.getElementById("classList");
        currentClass = classList.options[classList.selectedIndex].text;
        if (currentBbox !== null) {
            currentBbox.bbox.marked = false;
            currentBbox = null;
        }
        clearMultiPointAnnotations();
    };

    const listenClassSelect = () => {
        const classList = document.getElementById("classList");
        classList.addEventListener("change", () => {
            classListIndex = classList.selectedIndex;
            setCurrentClass();
        });
    };

    const listenBboxLoad = () => {
        const bboxesElement = document.getElementById("bboxes");
        const bboxesButton = document.getElementById("bboxesSelect");
        bboxesElement.addEventListener("click", () => {
            bboxesElement.value = "";
        });
        bboxesElement.addEventListener("change", (event) => {
            const files = event.target.files;
            if (files.length > 0) {
                resetBboxes();
                for (let i = 0; i < files.length; i++) {
                    const reader = new FileReader();
                    const extension = files[i].name.split(".").pop();
                    reader.addEventListener("load", () => {
                        if (extension === "txt" || extension === "xml" || extension === "json") {
                            storeBbox(files[i].name, reader.result);
                        } else {
                            const zip = new JSZip();
                            zip.loadAsync(reader.result)
                                .then((result) => {
                                    for (let filename in result.files) {
                                        result.file(filename).async("string")
                                            .then((text) => {
                                                storeBbox(filename, text);
                                            });
                                    }
                                });
                        }
                    });
                    if (extension === "txt" || extension === "xml" || extension === "json") {
                        reader.readAsText(files[i]);
                    } else {
                        reader.readAsArrayBuffer(event.target.files[i]);
                    }
                }
            }
            bboxesElement.value = "";
        });
    };

    const resetBboxes = () => {
        bboxes = {};
    };

    const storeBbox = (filename, text) => {
        // same storeBbox logic you had before
        let image = null;
        let bbox = null;
        const extension = filename.split(".").pop();
        if (extension === "txt" || extension === "xml") {
            for (let i = 0; i < extensions.length; i++) {
                const imageName = filename.replace(`.${extension}`, `.${extensions[i]}`);
                if (typeof images[imageName] !== "undefined") {
                    image = images[imageName];
                    if (typeof bboxes[imageName] === "undefined") {
                        bboxes[imageName] = {};
                    }
                    bbox = bboxes[imageName];
                    if (extension === "txt") {
                        const rows = text.split(/[\r\n]+/);
                        for (let i = 0; i < rows.length; i++) {
                            const cols = rows[i].split(" ");
                            cols[0] = parseInt(cols[0]);
                            for (let className in classes) {
                                if (classes[className] === cols[0]) {
                                    if (typeof bbox[className] === "undefined") {
                                        bbox[className] = [];
                                    }
                                    const width = cols[3] * image.width;
                                    const x = cols[1] * image.width - width * 0.5;
                                    const height = cols[4] * image.height;
                                    const y = cols[2] * image.height - height * 0.5;
                                    bbox[className].push({
                                        x: Math.floor(x),
                                        y: Math.floor(y),
                                        width: Math.floor(width),
                                        height: Math.floor(height),
                                        marked: false,
                                        class: className
                                    });
                                    break;
                                }
                            }
                        }
                    } else if (extension === "xml") {
                        const parser = new DOMParser();
                        const xmlDoc = parser.parseFromString(text, "text/xml");
                        const objects = xmlDoc.getElementsByTagName("object");
                        for (let i = 0; i < objects.length; i++) {
                            const objectName = objects[i].getElementsByTagName("name")[0].childNodes[0].nodeValue;
                            for (let className in classes) {
                                if (className === objectName) {
                                    if (typeof bbox[className] === "undefined") {
                                        bbox[className] = [];
                                    }
                                    const bndBox = objects[i].getElementsByTagName("bndbox")[0];
                                    const bndBoxX = bndBox.getElementsByTagName("xmin")[0].childNodes[0].nodeValue;
                                    const bndBoxY = bndBox.getElementsByTagName("ymin")[0].childNodes[0].nodeValue;
                                    const bndBoxMaxX = bndBox.getElementsByTagName("xmax")[0].childNodes[0].nodeValue;
                                    const bndBoxMaxY = bndBox.getElementsByTagName("ymax")[0].childNodes[0].nodeValue;
                                    bbox[className].push({
                                        x: parseInt(bndBoxX),
                                        y: parseInt(bndBoxY),
                                        width: parseInt(bndBoxMaxX) - parseInt(bndBoxX),
                                        height: parseInt(bndBoxMaxY) - parseInt(bndBoxY),
                                        marked: false,
                                        class: className
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            const json = JSON.parse(text);
            for (let i = 0; i < json.annotations.length; i++) {
                let imageName = null;
                let categoryName = null;
                for (let j = 0; j < json.images.length; j++) {
                    if (json.annotations[i].image_id === json.images[j].id) {
                        imageName = json.images[j].file_name;
                        if (typeof images[imageName] !== "undefined") {
                            image = images[imageName];
                            if (typeof bboxes[imageName] === "undefined") {
                                bboxes[imageName] = {};
                            }
                            bbox = bboxes[imageName];
                            break;
                        }
                    }
                }
                for (let j = 0; j < json.categories.length; j++) {
                    if (json.annotations[i].category_id === json.categories[j].id) {
                        categoryName = json.categories[j].name;
                        break;
                    }
                }
                for (let className in classes) {
                    if (className === categoryName) {
                        if (typeof bbox[className] === "undefined") {
                            bbox[className] = [];
                        }
                        const bboxX = json.annotations[i].bbox[0];
                        const bboxY = json.annotations[i].bbox[1];
                        const bboxWidth = json.annotations[i].bbox[2];
                        const bboxHeight = json.annotations[i].bbox[3];
                        bbox[className].push({
                            x: bboxX,
                            y: bboxY,
                            width: bboxWidth,
                            height: bboxHeight,
                            marked: false,
                            class: className
                        });
                        break;
                    }
                }
            }
        }
    };

    const listenBboxSave = () => {
        document.getElementById("saveBboxes").addEventListener("click", () => {
            const zip = new JSZip();
            for (let imageName in bboxes) {
                const image = images[imageName];
                if (!image) continue;
                const name = imageName.split(".");
                name[name.length - 1] = "txt";
                const result = [];
                for (let className in bboxes[imageName]) {
                    for (let i = 0; i < bboxes[imageName][className].length; i++) {
                        const bbox = bboxes[imageName][className][i];
                        const x = (bbox.x + bbox.width / 2) / image.width;
                        const y = (bbox.y + bbox.height / 2) / image.height;
                        const w = bbox.width / image.width;
                        const h = bbox.height / image.height;
                        result.push(`${classes[className]} ${x} ${y} ${w} ${h}`);
                    }
                }
                zip.file(name.join("."), result.join("\n"));
            }
            zip.generateAsync({ type: "blob" })
                .then((blob) => {
                    saveAs(blob, "bboxes_yolo.zip");
                });
        });
    };

    const listenKeyboard = () => {
        const imageList = document.getElementById("imageList");
        const classList = document.getElementById("classList");
        let modeSnapshot = null;

        document.addEventListener("keydown", (event) => {
            if (activeTab !== TAB_LABELING) {
                return;
            }
            const key = event.keyCode || event.charCode;

            if (!event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey && (key === 90 || event.key === "z" || event.key === "Z")) {
                if (!modeSnapshot) {
                    modeSnapshot = {
                        auto: autoMode,
                        sam: samMode,
                        point: pointMode,
                        multi: multiPointMode,
                    };
                    updateSamModeState(false, { preservePoints: true });
                    updateAutoModeState(false);
                }
                event.preventDefault();
                return;
            }

            if (key === 8 || (key === 46 && event.metaKey === true)) {
                if (currentBbox !== null) {
                    bboxes[currentImage.name][currentBbox.bbox.class].splice(currentBbox.index, 1);
                    currentBbox = null;
                    document.body.style.cursor = "default";
                }
                event.preventDefault();
            }
            // 'a' => toggle auto class
            if (key === 65 && !modeSnapshot) {
                updateAutoModeState(!autoMode);
                event.preventDefault();
            }
            // 's' => toggle SAM
            if (key === 83 && !modeSnapshot) {
                updateSamModeState(!samMode);
                event.preventDefault();
            }
            // 'd' => toggle SAM point mode
            if (key === 68 && !modeSnapshot) {
                if (!pointMode) {
                    if (!samMode) {
                        updateSamModeState(true);
                    }
                    updatePointModeState(true);
                } else {
                    updatePointModeState(false);
                }
                event.preventDefault();
            }
            // 'm' => toggle SAM multi-point mode
            if (key === 77 && !modeSnapshot) {
                if (!multiPointMode) {
                    if (!samMode) {
                        updateSamModeState(true);
                    }
                    updateMultiPointState(true);
                } else {
                    updateMultiPointState(false);
                }
                event.preventDefault();
            }
            // 'f' => add positive point
            if (!event.repeat && key === 70 && multiPointMode && !modeSnapshot) {
                addMultiPointAnnotation(1);
                event.preventDefault();
            }
            // 'g' => add negative point
            if (!event.repeat && key === 71 && multiPointMode && !modeSnapshot) {
                addMultiPointAnnotation(0);
                event.preventDefault();
            }
            // Enter => submit multi-point selection
            if (!event.repeat && key === 13 && multiPointMode && !modeSnapshot) {
                submitMultiPointSelection();
                event.preventDefault();
                return;
            }
            if (key === 37) {
                if (imageList.length > 1) {
                    imageList.options[imageListIndex].selected = false;
                    if (imageListIndex === 0) {
                        imageListIndex = imageList.length - 1;
                    } else {
                        imageListIndex--;
                    }
                    imageList.options[imageListIndex].selected = true;
                    imageList.selectedIndex = imageListIndex;
                    setCurrentImage(images[imageList.options[imageListIndex].innerHTML]);
                    document.body.style.cursor = "default";
                }
                event.preventDefault();
            }
            if (key === 39) {
                if (imageList.length > 1) {
                    imageList.options[imageListIndex].selected = false;
                    if (imageListIndex === imageList.length - 1) {
                        imageListIndex = 0;
                    } else {
                        imageListIndex++;
                    }
                    imageList.options[imageListIndex].selected = true;
                    imageList.selectedIndex = imageListIndex;
                    setCurrentImage(images[imageList.options[imageListIndex].innerHTML]);
                    document.body.style.cursor = "default";
                }
                event.preventDefault();
            }
            if (key === 38) {
                if (classList.length > 1) {
                    classList.options[classListIndex].selected = false;
                    if (classListIndex === 0) {
                        classListIndex = classList.length - 1;
                    } else {
                        classListIndex--;
                    }
                    classList.options[classListIndex].selected = true;
                    classList.selectedIndex = classListIndex;
                    setCurrentClass();
                }
                event.preventDefault();
            }
            if (key === 40) {
                if (classList.length > 1) {
                    classList.options[classListIndex].selected = false;
                    if (classListIndex === classList.length - 1) {
                        classListIndex = 0;
                    } else {
                        classListIndex++;
                    }
                    classList.options[classListIndex].selected = true;
                    classList.selectedIndex = classListIndex;
                    setCurrentClass();
                }
                event.preventDefault();
            }
        });

        document.addEventListener("keyup", (event) => {
            if (activeTab !== TAB_LABELING) {
                return;
            }
            const key = event.keyCode || event.charCode;
            if (modeSnapshot && (key === 90 || event.key === "z" || event.key === "Z")) {
                const snapshot = modeSnapshot;
                modeSnapshot = null;
                updateSamModeState(snapshot.sam);
                updateAutoModeState(snapshot.auto);
                updatePointModeState(snapshot.point);
                updateMultiPointState(snapshot.multi);
                event.preventDefault();
            }
        });
    };

    const resetCanvasPlacement = () => {
        scale = defaultScale;
        canvasX = 0;
        canvasY = 0;
        screenX = 0;
        screenY = 0;
        mouse.x = 0;
        mouse.y = 0;
        mouse.realX = 0;
        mouse.realY = 0;
        mouse.buttonL = 0;
        mouse.buttonR = 0;
        mouse.startRealX = 0;
        mouse.startRealY = 0;
    };

    const listenImageSearch = () => {
        document.getElementById("imageSearch").addEventListener("input", (event) => {
            const value = event.target.value;
            for (let imageName in images) {
                if (imageName.indexOf(value) !== -1) {
                    document.getElementById("imageList").selectedIndex = images[imageName].index;
                    setCurrentImage(images[imageName]);
                    break;
                }
            }
        });
    };

    function yieldToDom(delayMs = 50) {
        return new Promise((resolve) => {
            requestAnimationFrame(() => {
                setTimeout(resolve, delayMs);
            });
        });
    }

    async function extractBase64ForImage(imgObj) {
        const offCanvas = document.createElement("canvas");
        offCanvas.width = imgObj.width;
        offCanvas.height = imgObj.height;
        const ctx = offCanvas.getContext("2d");
        ctx.drawImage(imgObj.object, 0, 0, imgObj.width, imgObj.height);
        const dataUrl = offCanvas.toDataURL("image/jpeg");
        return dataUrl.split(",")[1];
    }

        /****************************************************
     * listenImageCrop - single-image-at-a-time approach
     ****************************************************/
    async function listenImageCrop() {
      setupTabNavigation();
      const btn = document.getElementById("cropImages");
      btn.addEventListener("click", async () => {
        const imageNames = Object.keys(bboxes);
        if (!imageNames.length) {
          alert("No bounding boxes to crop.");
          return;
        }

        const progressModal = showProgressModal("Initializing crop job...");
        document.body.style.cursor = "wait";

        try {
          // 1) Start the server-side job
          let resp = await fetch("http://localhost:8000/crop_zip_init", { method: "POST" });
          if (!resp.ok) {
            throw new Error("crop_zip_init failed: " + resp.status);
          }
          const { jobId } = await resp.json();
          console.log("Got jobId:", jobId);

          // 2) Single-image loop
          let count = 0;
          for (const imgName of imageNames) {
            count++;
            progressModal.update(`Processing ${count} / ${imageNames.length}: ${imgName}`);

            const imgData = images[imgName];
            if (!imgData) continue;

            // Load image if not loaded
            if (!imgData.object) {
              await loadImageObject(imgData);
            }

            // Gather bounding boxes
            const rawBoxes = bboxes[imgName];
            if (!rawBoxes) {
              // No bboxes at all for this image
              imgData.object = null;
              await yieldToDom(10);
              continue;
            }

            // Flatten bounding boxes & clamp
            const allBbs = [];
            for (const className in rawBoxes) {
              rawBoxes[className].forEach(bbox => {
                const copy = { ...bbox };
                const valid = clampBbox(copy, imgData.width, imgData.height);
                if (valid) {
                  allBbs.push({
                    className, 
                    x: copy.x,
                    y: copy.y,
                    width: copy.width,
                    height: copy.height
                  });
                }
              });
            }
            if (!allBbs.length) {
              // no valid bounding boxes => skip
              console.log("No valid bounding boxes for", imgName, " => skipping");
              imgData.object = null;
              await yieldToDom(10);
              continue;
            }

            // Convert to base64
            const base64Img = await extractBase64ForImage(imgData);

            // 3) Send just this one image
            // If your server wants different keys, rename them here
            const body = {
              images: [
                {
                  image_base64: base64Img,
                  originalName: imgName,
                  bboxes: allBbs
                }
              ]
            };
            // Debug log
            console.log("Sending:", JSON.stringify(body, null, 2));

            resp = await fetch(`http://localhost:8000/crop_zip_chunk?jobId=${jobId}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body)
            });
            if (!resp.ok) {
              throw new Error(`crop_zip_chunk failed: ${resp.status}`);
            }

            // free memory
            imgData.object = null;
            await yieldToDom(10);
          }

          // 4) Finalize
          progressModal.update("Finalizing crop_zip_finalize...");
          resp = await fetch(`http://localhost:8000/crop_zip_finalize?jobId=${jobId}`);
          if (!resp.ok) {
            throw new Error("crop_zip_finalize failed: " + resp.status);
          }
          const blob = await resp.blob();
          saveAs(blob, "crops.zip");
          alert("Done! crops.zip downloaded.");
        } catch (err) {
          console.error(err);
          alert("Crop & Save failed: " + err);
        } finally {
          progressModal.close();
          document.body.style.cursor = "default";
        }
      });
    }

    // Example clamp function
    function clampBbox(bbox, imgW, imgH) {
      // ensure x, y >= 0
      bbox.x = Math.max(0, bbox.x);
      bbox.y = Math.max(0, bbox.y);

      // ensure x + w <= imgW
      if (bbox.x > imgW) return false;
      if (bbox.y > imgH) return false;

      const maxW = imgW - bbox.x;
      const maxH = imgH - bbox.y;
      bbox.width = Math.min(bbox.width, maxW);
      bbox.height = Math.min(bbox.height, maxH);

      if (bbox.width <= 0 || bbox.height <= 0) return false;
      return true;
    }

    /**
     * Helper function to load an image from its File object (imgData.meta)
     * and store the resulting <img> in imgData.object.
     */
    function loadImageObject(imgData) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const im = new Image();
                im.onload = () => {
                    imgData.object = im;
                    resolve();
                };
                im.onerror = reject;
                im.src = reader.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(imgData.meta);
        });
    }

    function chunkArray(array, size) {
        const result = [];
        for (let i = 0; i < array.length; i += size) {
            result.push(array.slice(i, i + size));
        }
        return result;
    }

    // Enhanced showProgressModal that can update text
    function showProgressModal(initialText = "") {
        const overlay = document.createElement("div");
        overlay.style.position = "fixed";
        overlay.style.top = "0";
        overlay.style.left = "0";
        overlay.style.width = "100%";
        overlay.style.height = "100%";
        overlay.style.backgroundColor = "rgba(0, 0, 0, 0.75)";
        overlay.style.zIndex = "9999";
        overlay.style.display = "flex";
        overlay.style.alignItems = "center";
        overlay.style.justifyContent = "center";
        overlay.style.flexDirection = "column";

        const textDiv = document.createElement("div");
        textDiv.style.color = "#fff";
        textDiv.style.fontSize = "20px";
        textDiv.style.marginBottom = "10px";
        textDiv.textContent = initialText;

        overlay.appendChild(textDiv);
        document.body.appendChild(overlay);

        return {
            update(msg) {
                textDiv.textContent = msg;
            },
            close() {
                document.body.removeChild(overlay);
            }
        };
    }

})();
