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
    const slotPreloadControllers = { next: null, previous: null };
    const slotPreloadPromises = new Map();
    const slotLoadingIndicators = new Map();
    let samPreloadQueuedKey = null;
    let latestSlotStatuses = [];
    let samSlotStatusTimer = null;
    let samSlotStatusPending = false;
    let samSlotStatusNeedsRefresh = false;
    const SAM_SLOT_STATUS_DEBOUNCE_MS = 600;
    let samSlotsEnabled = false;
    let samSlotsSupportChecked = false;
    let samPredictorBudget = 3;
    let predictorTabInitialized = false;
    let predictorRefreshTimer = null;
    let predictorRefreshInFlight = false;
    const PREDICTOR_REFRESH_INTERVAL_MS = 5000;
    let predictorSettings = {
        maxPredictors: 3,
        minPredictors: 1,
        maxSupportedPredictors: 3,
        activePredictors: 3,
        loadedPredictors: 0,
        processRamMb: 0,
        totalRamMb: 0,
        availableRamMb: 0,
        imageRamMb: 0,
    };

    let imagesSelectButton = null;
    let classesSelectButton = null;
    let bboxesSelectButton = null;
    let bboxesFolderSelectButton = null;

    let samStatusEl = null;
    let samStatusTimer = null;
    let samStatusMessageToken = 0;
    let samJobSequence = 0;
    let samCancelVersion = 0;
    const samActiveJobs = new Map();
    let imageListSelectionLock = 0;
    let imageLoadInProgress = false;
    let imageLoadPromise = null;
    const tweakPreserveSet = new Set();
    let magicTweakRunning = false;

    const multiPointColors = {
        positive: { stroke: "#2ecc71", fill: "rgba(46, 204, 113, 0.35)" },
        negative: { stroke: "#e74c3c", fill: "rgba(231, 76, 60, 0.35)" },
    };

    const DEFAULT_API_ROOT = "http://localhost:8000";
    const API_STORAGE_KEY = "tator.apiRoot";
    let API_ROOT = loadStoredApiRoot();
    const TAB_LABELING = "labeling";
    const TAB_TRAINING = "training";
    const TAB_QWEN_TRAIN = "qwen-train";
    const TAB_ACTIVE = "active";
    const TAB_QWEN = "qwen";
    const TAB_PREDICTORS = "predictors";
    const TAB_SETTINGS = "settings";

    function loadStoredApiRoot() {
        try {
            const saved = localStorage.getItem(API_STORAGE_KEY);
            const normalized = normalizeApiRoot(saved);
            return normalized || DEFAULT_API_ROOT;
        } catch (error) {
            console.debug("Failed to read stored API root", error);
            return DEFAULT_API_ROOT;
        }
    }

    function normalizeApiRoot(value) {
        if (!value) {
            return null;
        }
        let trimmed = String(value).trim();
        if (!trimmed) {
            return null;
        }
        if (!/^https?:\/\//i.test(trimmed)) {
            trimmed = `http://${trimmed}`;
        }
        trimmed = trimmed.replace(/\/+$/, "");
        return trimmed;
    }

    let activeTab = TAB_LABELING;
    let trainingUiInitialized = false;
    let activeUiInitialized = false;
    let loadedClassList = [];
    const INGEST_PHASE_LABELS = {
        images: "Loading images",
        bboxes: "Importing bboxes",
    };

    function ensureIngestElements() {
        if (!ingestProgressState.element) {
            ingestProgressState.element = document.getElementById("ingestProgress");
            ingestProgressState.labelEl = document.getElementById("ingestProgressLabel");
            ingestProgressState.detailEl = document.getElementById("ingestProgressDetail");
        }
    }

    function startIngestProgress({ phase, total = 0, extraLabel = null }) {
        ensureIngestElements();
        if (!ingestProgressState.element) {
            return;
        }
        ingestProgressState.phase = phase;
        ingestProgressState.total = Math.max(0, Number(total) || 0);
        ingestProgressState.completed = 0;
        ingestProgressState.extraLabel = extraLabel;
        ingestProgressState.extraValue = 0;
        ingestProgressState.visible = true;
        renderIngestProgress();
    }

    function renderIngestProgress() {
        ensureIngestElements();
        if (!ingestProgressState.element) {
            return;
        }
        if (!ingestProgressState.visible) {
            ingestProgressState.element.classList.remove("visible");
            return;
        }
        const label = INGEST_PHASE_LABELS[ingestProgressState.phase] || "Loading";
        const total = ingestProgressState.total;
        const completed = total ? Math.min(ingestProgressState.completed, total) : ingestProgressState.completed;
        let detail = total ? `${completed}/${total}` : `${completed}`;
        if (ingestProgressState.extraLabel) {
            detail += ` • ${ingestProgressState.extraLabel}: ${ingestProgressState.extraValue}`;
        }
        ingestProgressState.labelEl.textContent = label;
        ingestProgressState.detailEl.textContent = detail;
        ingestProgressState.element.classList.add("visible");
    }

    function incrementIngestProgress(delta = 1) {
        if (!ingestProgressState.visible) {
            return;
        }
        ingestProgressState.completed = Math.max(0, ingestProgressState.completed + delta);
        renderIngestProgress();
    }

    function adjustIngestTotal(delta) {
        if (!ingestProgressState.visible || !delta) {
            return;
        }
        ingestProgressState.total = Math.max(0, ingestProgressState.total + delta);
        renderIngestProgress();
    }

    function incrementIngestExtra(delta = 1) {
        if (!ingestProgressState.visible || !ingestProgressState.extraLabel) {
            return;
        }
        ingestProgressState.extraValue = Math.max(0, ingestProgressState.extraValue + delta);
        renderIngestProgress();
    }

    function stopIngestProgress() {
        ensureIngestElements();
        if (!ingestProgressState.element) {
            return;
        }
        ingestProgressState.visible = false;
        ingestProgressState.phase = null;
        ingestProgressState.total = 0;
        ingestProgressState.completed = 0;
        ingestProgressState.extraValue = 0;
        ingestProgressState.element.classList.remove("visible");
    }

    function noteImportedBbox(count = 1) {
        if (!bboxImportCounterActive) {
            return;
        }
        incrementIngestExtra(count);
    }

    function ensureBackgroundLoadElements() {
        if (backgroundLoadModal.element) {
            return;
        }
        backgroundLoadModal.element = document.getElementById("backgroundLoadModal");
        backgroundLoadModal.dismissBtn = document.getElementById("backgroundLoadDismiss");
        if (backgroundLoadModal.dismissBtn) {
            backgroundLoadModal.dismissBtn.addEventListener("click", () => hideBackgroundLoadModal());
        }
        const backdrop = backgroundLoadModal.element?.querySelector(".modal__backdrop");
        if (backdrop) {
            backdrop.addEventListener("click", () => hideBackgroundLoadModal());
        }
    }

    function showBackgroundLoadModal(message = null) {
        ensureBackgroundLoadElements();
        if (!backgroundLoadModal.element) {
            return;
        }
        const msgEl = document.getElementById("backgroundLoadMessage");
        if (msgEl && message) {
            msgEl.textContent = message;
        }
        backgroundLoadModal.visible = true;
        backgroundLoadModal.element.classList.add("visible");
        backgroundLoadModal.element.setAttribute("aria-hidden", "false");
    }

    function hideBackgroundLoadModal() {
        if (!backgroundLoadModal.element) {
            return;
        }
        backgroundLoadModal.visible = false;
        backgroundLoadModal.element.classList.remove("visible");
        backgroundLoadModal.element.setAttribute("aria-hidden", "true");
    }

    function ensureTrainingPackagingElements() {
        if (trainingPackagingModal.element) {
            return;
        }
        trainingPackagingModal.element = document.getElementById("trainingPackagingModal");
        trainingPackagingModal.summaryEl = document.getElementById("trainingPackagingStats");
        trainingPackagingModal.etaEl = document.getElementById("trainingPackagingEta");
        trainingPackagingModal.elapsedEl = document.getElementById("trainingPackagingElapsed");
        trainingPackagingModal.hintEl = document.getElementById("trainingPackagingHint");
        trainingPackagingModal.progressLabel = document.getElementById("trainingPackagingProgressText");
        trainingPackagingModal.progressFill = document.getElementById("trainingPackagingProgressFill");
        trainingPackagingModal.dismissBtn = document.getElementById("trainingPackagingDismiss");
        if (trainingPackagingModal.dismissBtn) {
            trainingPackagingModal.dismissBtn.addEventListener("click", () => hideTrainingPackagingModal());
        }
        const backdrop = trainingPackagingModal.element?.querySelector(".modal__backdrop");
        if (backdrop) {
            backdrop.addEventListener("click", () => hideTrainingPackagingModal());
        }
    }

    function showTrainingPackagingModal(stats, options = {}) {
        ensureTrainingPackagingElements();
        if (!trainingPackagingModal.element) {
            return;
        }
        if (trainingPackagingModal.timerId) {
            clearInterval(trainingPackagingModal.timerId);
            trainingPackagingModal.timerId = null;
        }
        const {
            hintText = null,
            progressText = "Preparing files…",
            indeterminate = true,
        summaryText = null,
        } = options;
        const imageSummary = stats
            ? `${stats.imageCount} image${stats.imageCount === 1 ? "" : "s"} (${formatBytes(stats.imageBytes)})`
            : "";
        const labelSummary = stats
            ? `${stats.labelCount} label file${stats.labelCount === 1 ? "" : "s"} (${formatBytes(stats.labelBytes)})`
            : "";
        const totalSummary = stats ? `${stats.totalFiles} files ≈ ${formatBytes(stats.totalBytes)}` : null;
        if (trainingPackagingModal.summaryEl) {
            if (summaryText) {
                trainingPackagingModal.summaryEl.textContent = summaryText;
            } else if (stats) {
                trainingPackagingModal.summaryEl.textContent = `${imageSummary} + ${labelSummary} (${totalSummary})`;
            } else {
                trainingPackagingModal.summaryEl.textContent = "Packaging dataset…";
            }
        }
        if (trainingPackagingModal.etaEl) {
            if (stats && Number.isFinite(stats.estimatedSeconds) && stats.estimatedSeconds > 0) {
                trainingPackagingModal.etaEl.textContent = `Estimated upload: ${formatDurationPrecise(stats.estimatedSeconds)} (${describeDurationRange(stats.estimatedSeconds)})`;
            } else {
                trainingPackagingModal.etaEl.textContent = "Estimating upload time…";
            }
        }
        if (trainingPackagingModal.progressLabel) {
            trainingPackagingModal.progressLabel.textContent = progressText;
        }
        if (trainingPackagingModal.progressFill) {
            trainingPackagingModal.progressFill.style.width = indeterminate ? "200%" : "0%";
            trainingPackagingModal.progressFill.classList.toggle("is-indeterminate", indeterminate);
        }
        trainingPackagingModal.indeterminate = indeterminate;
        if (trainingPackagingModal.elapsedEl) {
            trainingPackagingModal.elapsedEl.textContent = "Elapsed: 0s";
        }
        if (trainingPackagingModal.hintEl && hintText) {
            trainingPackagingModal.hintEl.textContent = hintText;
        } else if (trainingPackagingModal.hintEl && !hintText) {
            trainingPackagingModal.hintEl.textContent = "Keep this tab open while we stage files and upload them to the server. Larger datasets can take a few minutes.";
        }
        trainingPackagingModal.startedAt = performance.now();
        trainingPackagingModal.visible = true;
        trainingPackagingModal.element.classList.add("visible");
        trainingPackagingModal.element.setAttribute("aria-hidden", "false");
        trainingPackagingModal.timerId = window.setInterval(() => {
            updateTrainingPackagingElapsed();
        }, 500);
    }

    function updateTrainingPackagingElapsed() {
        if (!trainingPackagingModal.visible || !trainingPackagingModal.elapsedEl) {
            return;
        }
        const elapsedSeconds = Math.max(0, (performance.now() - trainingPackagingModal.startedAt) / 1000);
        trainingPackagingModal.elapsedEl.textContent = `Elapsed: ${formatDurationPrecise(elapsedSeconds)}`;
    }

    function updateTrainingPackagingProgress(percent, text) {
        ensureTrainingPackagingElements();
        if (!trainingPackagingModal.visible) {
            return;
        }
        if (typeof percent === "number" && trainingPackagingModal.progressFill) {
            const clamped = Math.max(0, Math.min(100, percent));
            trainingPackagingModal.progressFill.classList.remove("is-indeterminate");
            trainingPackagingModal.progressFill.style.width = `${clamped}%`;
            trainingPackagingModal.indeterminate = false;
        }
        if (text && trainingPackagingModal.progressLabel) {
            trainingPackagingModal.progressLabel.textContent = text;
        }
    }

    function hideTrainingPackagingModal() {
        if (!trainingPackagingModal.element) {
            return;
        }
        trainingPackagingModal.visible = false;
        trainingPackagingModal.element.classList.remove("visible");
        trainingPackagingModal.element.setAttribute("aria-hidden", "true");
        if (trainingPackagingModal.timerId) {
            clearInterval(trainingPackagingModal.timerId);
            trainingPackagingModal.timerId = null;
        }
        if (trainingPackagingModal.progressFill) {
            trainingPackagingModal.progressFill.classList.add("is-indeterminate");
            trainingPackagingModal.progressFill.style.width = "200%";
        }
        if (trainingPackagingModal.progressLabel) {
            trainingPackagingModal.progressLabel.textContent = "Preparing…";
        }
    }

    function ensureTaskQueueElement() {
        if (!taskQueueState.element) {
            taskQueueState.element = document.getElementById("taskQueue");
        }
        return taskQueueState.element;
    }

    function enqueueTask({ kind, imageName, detail }) {
        const container = ensureTaskQueueElement();
        if (!container) {
            return null;
        }
        const group = TASK_GROUP_MAP[kind] || kind || "misc";
        const entry = {
            id: ++taskQueueState.counter,
            kind: kind || "sam",
            group,
            imageName: imageName || null,
            detail: detail || null,
            timestamp: Date.now(),
        };
        taskQueueState.items.push(entry);
        renderTaskQueue();
        return entry.id;
    }

    function completeTask(taskId) {
        if (!taskId) {
            return;
        }
        const idx = taskQueueState.items.findIndex((item) => item.id === taskId);
        if (idx !== -1) {
            taskQueueState.items.splice(idx, 1);
            renderTaskQueue();
        }
    }

    function clearTaskQueue(predicate, { statusMessage = null } = {}) {
        if (!taskQueueState.items.length) {
            return;
        }
        if (typeof predicate !== "function") {
            taskQueueState.items = [];
        } else {
            taskQueueState.items = taskQueueState.items.filter((item) => !predicate(item));
        }
        if (statusMessage) {
            setSamStatus(statusMessage, { variant: "warn", duration: 4000 });
        }
        renderTaskQueue();
    }

    function renderTaskQueue() {
        const container = ensureTaskQueueElement();
        if (!container) {
            return;
        }
        if (!taskQueueState.items.length) {
            container.innerHTML = "";
            container.classList.remove("visible");
            return;
        }
        const summary = new Map();
        taskQueueState.items.forEach((item) => {
            const group = item.group || item.kind || "misc";
            summary.set(group, (summary.get(group) || 0) + 1);
        });
        const fragments = Array.from(summary.entries()).map(([group, count]) => {
            const label = (TASK_GROUP_LABELS[group] || group).toLowerCase();
            const noun = count === 1 ? "task" : "tasks";
            return `<div class="task-queue__entry"><span class="task-queue__label">${count} ${label} ${noun} pending</span></div>`;
        });
        container.innerHTML = fragments.join("");
        container.classList.add("visible");
    }

    function shortenName(name) {
        if (!name) {
            return "—";
        }
        return name.length > 10 ? `${name.slice(0, 10)}…` : name;
    }

    function ensureBatchTweakElements() {
        if (!batchTweakElements.modal) {
            batchTweakElements.modal = document.getElementById("batchTweakModal");
            batchTweakElements.backdrop = batchTweakElements.modal?.querySelector(".modal__backdrop");
            batchTweakElements.confirm = document.getElementById("batchTweakConfirm");
            batchTweakElements.cancel = document.getElementById("batchTweakCancel");
            batchTweakElements.classLabel = document.getElementById("batchTweakClass");
            if (batchTweakElements.confirm) {
                batchTweakElements.confirm.addEventListener("click", () => {
                    closeBatchTweakModal();
                    runBatchTweakForCurrentCategory().catch((err) => {
                        console.error("Batch tweak failed", err);
                        setSamStatus(`Batch tweak failed: ${err.message || err}`, { variant: "error", duration: 5000 });
                    });
                });
            }
            if (batchTweakElements.cancel) {
                batchTweakElements.cancel.addEventListener("click", () => {
                    closeBatchTweakModal();
                });
            }
            if (batchTweakElements.backdrop) {
                batchTweakElements.backdrop.addEventListener("click", () => closeBatchTweakModal());
            }
            document.addEventListener("keydown", (event) => {
                if (event.key === "Escape" && batchTweakElements.modal?.classList.contains("visible")) {
                    closeBatchTweakModal();
                }
            });
        }
    }

    function openBatchTweakModal() {
        ensureBatchTweakElements();
        if (!batchTweakElements.modal) {
            return;
        }
        if (!currentClass) {
            setSamStatus("Select a class in the list before batch tweaking", { variant: "warn", duration: 3000 });
            return;
        }
        if (batchTweakRunning) {
            setSamStatus("Batch tweak already running", { variant: "info", duration: 2500 });
            return;
        }
        if (!currentImage || !currentImage.name || !currentClass) {
            setSamStatus("Choose an image and class before batch tweaking", { variant: "warn", duration: 3000 });
            return;
        }
        const bucket = bboxes[currentImage.name]?.[currentClass] || [];
        if (!bucket.length) {
            setSamStatus("No bboxes available for this class", { variant: "warn", duration: 3000 });
            return;
        }
        if (!samMode) {
            setSamStatus("Enable SAM mode to batch tweak", { variant: "warn", duration: 3000 });
            return;
        }
        if (batchTweakElements.classLabel) {
            batchTweakElements.classLabel.textContent = `${currentClass} (${bucket.length})`;
        }
        batchTweakElements.modal.classList.add("visible");
        batchTweakElements.modal.setAttribute("aria-hidden", "false");
    }

    function closeBatchTweakModal() {
        if (!batchTweakElements.modal) {
            return;
        }
        batchTweakElements.modal.classList.remove("visible");
        batchTweakElements.modal.setAttribute("aria-hidden", "true");
    }

    function requestBatchTweakModal() {
        if (!currentClass) {
            setSamStatus("Select a class in the list before batch tweaking", { variant: "warn", duration: 3000 });
            return;
        }
        openBatchTweakModal();
    }

    function handleXHotkeyPress() {
        if (xHotkeyTimeoutId) {
            clearTimeout(xHotkeyTimeoutId);
            xHotkeyTimeoutId = null;
            requestBatchTweakModal();
            return;
        }
        xHotkeyTimeoutId = window.setTimeout(() => {
            xHotkeyTimeoutId = null;
            handleMagicTweakHotkey().catch((err) => {
                console.debug("Magic tweak hotkey failed", err);
            });
        }, DOUBLE_TAP_WINDOW_MS);
    }

    const ingestProgressState = {
        element: null,
        labelEl: null,
        detailEl: null,
        phase: null,
        total: 0,
        completed: 0,
        extraLabel: null,
        extraValue: 0,
        visible: false,
    };
    let bboxImportCounterActive = false;
    const DOUBLE_TAP_WINDOW_MS = 260;
    let xHotkeyTimeoutId = null;
    let batchTweakRunning = false;
    const batchTweakElements = {
        modal: null,
        backdrop: null,
        confirm: null,
        cancel: null,
        classLabel: null,
    };
    const backgroundLoadModal = {
        element: null,
        dismissBtn: null,
        visible: false,
        decimalsTotal: 0,
        decimalsDone: 0,
    };
    const trainingPackagingModal = {
        element: null,
        summaryEl: null,
        etaEl: null,
        elapsedEl: null,
        hintEl: null,
        progressLabel: null,
        progressFill: null,
        dismissBtn: null,
        visible: false,
        startedAt: 0,
        timerId: null,
        indeterminate: true,
    };
    const taskQueueState = {
        element: null,
        items: [],
        counter: 0,
    };
    const TASK_GROUP_MAP = {
        "sam-bbox": "bbox",
        "sam-bbox-auto": "bbox",
        "sam-point": "bbox",
        "sam-point-auto": "bbox",
        "sam-multipoint": "bbox",
        "sam-multipoint-auto": "bbox",
        "sam-batch": "bbox",
        "sam-preload": "preload",
        "sam-activate": "preload",
        sam: "bbox",
        "clip-auto": "clip",
    };

    const TASK_GROUP_LABELS = {
        preload: "SAM preloads",
        bbox: "bbox adjustments",
        clip: "CLIP tasks",
    };

    const tabElements = {
        labelingButton: null,
        trainingButton: null,
        qwenTrainButton: null,
        activeButton: null,
        qwenButton: null,
        predictorsButton: null,
        settingsButton: null,
        labelingPanel: null,
        trainingPanel: null,
        qwenTrainPanel: null,
        activePanel: null,
        qwenPanel: null,
        predictorsPanel: null,
        settingsPanel: null,
    };


    const predictorElements = {
        countInput: null,
        applyButton: null,
        message: null,
        activeCount: null,
        loadedCount: null,
        processRam: null,
        imageRam: null,
        systemFreeRam: null,
    };

    const settingsElements = {
        apiInput: null,
        applyButton: null,
        testButton: null,
        status: null,
    };

    const qwenElements = {
        statusLabel: null,
        itemsInput: null,
        manualPrompt: null,
        imageTypeInput: null,
        extraContextInput: null,
        advancedToggle: null,
        advancedPanel: null,
        promptType: null,
        classSelect: null,
        maxResults: null,
        runButton: null,
    };
    const DEFAULT_QWEN_METADATA = {
        id: "default",
        label: "Base Qwen",
        dataset_context: "",
        classes: [],
        system_prompt: "",
    };

    const qwenModelElements = {
        status: null,
        list: null,
        details: null,
    };
    let qwenAvailable = false;
    let qwenRequestActive = false;
    let qwenClassOverride = false;
    let qwenAdvancedVisible = false;
    const qwenModelState = {
        models: [],
        activeId: "default",
        activeMetadata: DEFAULT_QWEN_METADATA,
    };

    let settingsUiInitialized = false;

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

    const qwenTrainElements = {
        runNameInput: null,
        contextInput: null,
        modelIdInput: null,
        systemPromptInput: null,
        systemPromptNoiseInput: null,
        acceleratorSelect: null,
        loraRadios: null,
        batchSizeInput: null,
        epochsInput: null,
        lrInput: null,
        accumulateInput: null,
        loraRankInput: null,
        loraAlphaInput: null,
        loraDropoutInput: null,
        patienceInput: null,
        sampleButton: null,
        sampleCanvas: null,
        samplePrompt: null,
        sampleExpected: null,
        sampleMessage: null,
        sampleMeta: null,
        startButton: null,
        cancelButton: null,
        progressFill: null,
        statusText: null,
        message: null,
        summary: null,
        log: null,
        historyContainer: null,
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

    const qwenTrainState = {
        activeJobId: null,
        pollHandle: null,
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

    function formatBytes(bytes, digits = 1) {
        if (!Number.isFinite(bytes) || bytes <= 0) {
            return "0 B";
        }
        const units = ["B", "KB", "MB", "GB", "TB"];
        let value = bytes;
        let unitIndex = 0;
        while (value >= 1024 && unitIndex < units.length - 1) {
            value /= 1024;
            unitIndex += 1;
        }
        const precision = value >= 100 ? 0 : digits;
        return `${value.toFixed(precision)} ${units[unitIndex]}`;
    }

    function formatDurationPrecise(seconds) {
        if (!Number.isFinite(seconds) || seconds <= 0) {
            return "0s";
        }
        const totalSeconds = Math.floor(seconds);
        const mins = Math.floor(totalSeconds / 60);
        const hrs = Math.floor(mins / 60);
        const secs = totalSeconds % 60;
        const minutesPart = mins % 60;
        const parts = [];
        if (hrs > 0) {
            parts.push(`${hrs}h`);
        }
        if (minutesPart > 0 || hrs > 0) {
            parts.push(`${minutesPart}m`);
        }
        parts.push(`${secs}s`);
        return parts.join(" ");
    }

    function describeDurationRange(seconds) {
        if (!Number.isFinite(seconds) || seconds <= 0) {
            return "under a minute";
        }
        if (seconds < 60) {
            return "under a minute";
        }
        if (seconds < 180) {
            return "a few minutes";
        }
        if (seconds < 3600) {
            const mins = Math.round(seconds / 60);
            return `${mins} minute${mins === 1 ? "" : "s"}`;
        }
        const hours = seconds / 3600;
        if (hours < 10) {
            return `${hours.toFixed(1)} hours`;
        }
        return `${Math.round(hours)} hours`;
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

function setQwenTrainMessage(text, variant = null) {
    if (!qwenTrainElements.message) {
        return;
    }
    qwenTrainElements.message.textContent = text || "";
    qwenTrainElements.message.classList.remove("error", "warn", "success");
    if (variant) {
        qwenTrainElements.message.classList.add(variant);
    }
}

function readNumberInput(input, { integer = false } = {}) {
    if (!input) {
        return undefined;
    }
    const raw = String(input.value ?? "").trim();
    if (!raw) {
        return undefined;
    }
    const parsed = integer ? parseInt(raw, 10) : parseFloat(raw);
    return Number.isFinite(parsed) ? parsed : undefined;
}

function getSelectedQwenLoraMode() {
    const selected = document.querySelector('input[name="qwenLoraMode"]:checked');
    return selected ? selected.value : "qlora";
}

function shuffleArray(input) {
    const arr = [...input];
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

function sanitizeDatasetFilename(name) {
    return (name || "image").replace(/[\\/]/g, "_");
}

function makeUniqueFilename(baseName, usageMap) {
    const safeBase = sanitizeDatasetFilename(baseName);
    const count = usageMap.get(safeBase) || 0;
    usageMap.set(safeBase, count + 1);
    if (count === 0) {
        return safeBase;
    }
    const dotIndex = safeBase.lastIndexOf(".");
    if (dotIndex !== -1) {
        const stem = safeBase.slice(0, dotIndex);
        const ext = safeBase.slice(dotIndex);
        return `${stem}_${count}${ext}`;
    }
    return `${safeBase}_${count}`;
}

async function ensureImageDimensions(imageRecord) {
    if (imageRecord.width && imageRecord.height) {
        return;
    }
    await loadImageObject(imageRecord);
    const width = imageRecord.object?.naturalWidth || imageRecord.object?.width || imageRecord.width;
    const height = imageRecord.object?.naturalHeight || imageRecord.object?.height || imageRecord.height;
    if (!width || !height) {
        throw new Error(`Unable to read dimensions for ${imageRecord.meta?.name || "image"}`);
    }
    imageRecord.width = width;
    imageRecord.height = height;
}

function buildQwenInstruction(contextText, classNames) {
    const parts = [];
    if (contextText) {
        parts.push(`This image shows ${contextText}.`);
    }
    if (classNames.length) {
        parts.push(`Objects of interest: ${classNames.join(", ")}.`);
    }
    return parts.join(" ").trim();
}

function buildDetectionRecords(imageName, imageRecord) {
    const width = imageRecord.width || 0;
    const height = imageRecord.height || 0;
    const buckets = bboxes[imageName] || {};
    const records = [];
    for (const className of Object.keys(buckets)) {
        const bucket = buckets[className] || [];
        bucket.forEach((bbox) => {
            if (!bbox) {
                return;
            }
            const copy = {
                x: bbox.x,
                y: bbox.y,
                width: bbox.width,
                height: bbox.height,
            };
            const valid = clampBbox(copy, width, height);
            if (!valid) {
                return;
            }
            const x1 = Math.round(copy.x);
            const y1 = Math.round(copy.y);
            const x2 = Math.round(copy.x + copy.width);
            const y2 = Math.round(copy.y + copy.height);
            const cx = Math.round(copy.x + copy.width / 2);
            const cy = Math.round(copy.y + copy.height / 2);
            records.push({
                label: className,
                bbox: [x1, y1, x2, y2],
                point: [cx, cy],
            });
        });
    }
    return records;
}

function chooseQwenSampleLabelSet(detections) {
    const labels = Array.from(new Set((detections || []).map((det) => (det.label || "").trim()).filter(Boolean))).sort();
    if (!labels.length) {
        return { labels: [], mode: "all" };
    }
    if (labels.length === 1) {
        return Math.random() < 0.5 ? { labels, mode: "single" } : { labels, mode: "all" };
    }
    const roll = Math.random();
    if (roll < 0.34) {
        return { labels, mode: "all" };
    }
    if (roll < 0.67) {
        return { labels: [labels[Math.floor(Math.random() * labels.length)]], mode: "single" };
    }
    const subsetSize = Math.max(2, Math.floor(Math.random() * labels.length) + 1);
    const shuffled = shuffleArray(labels.slice());
    return { labels: shuffled.slice(0, Math.min(subsetSize, labels.length)).sort(), mode: "subset" };
}

function filterDetectionsForLabels(detections, labels, mode) {
    if (!Array.isArray(detections) || !detections.length || !labels || !labels.length || mode === "all") {
        return detections || [];
    }
    const labelSet = new Set(labels.map((label) => label.trim()).filter(Boolean));
    return detections.filter((det) => labelSet.has((det.label || "").trim()));
}

function buildQwenSampleUserPrompt(context, labels, mode, type) {
    const parts = [];
    if (context) {
        parts.push(context);
    }
    if (labels && labels.length) {
        if (mode === "single") {
            parts.push(`Focus only on the class '${labels[0]}'.`);
        } else if (mode === "subset") {
            parts.push(`Focus only on these classes: ${labels.join(", ")}.`);
        } else {
            parts.push(`Return detections for these classes: ${labels.join(", ")}.`);
        }
    } else {
        parts.push("Return detections for every labeled object.");
    }
    if (type === "bbox") {
        parts.push("Return a JSON object named \"detections\". Each detection must include \"label\" and \"bbox\" as [x1,y1,x2,y2] pixel coordinates (integers). If nothing is present, respond with {\"detections\": []}. Respond with JSON only.");
    } else {
        parts.push("Return a JSON object named \"detections\". Each detection must include \"label\" and \"point\" as [x,y] pixel coordinates near the object center. If nothing is present, respond with {\"detections\": []}. Respond with JSON only.");
    }
    return parts.filter(Boolean).join(" ").trim();
}

function pickQwenValidationSet(imageNames) {
    const shuffled = shuffleArray(imageNames);
    if (!shuffled.length) {
        return new Set();
    }
    let valCount = Math.max(1, Math.round(shuffled.length * 0.2));
    if (valCount >= shuffled.length && shuffled.length > 1) {
        valCount = Math.max(1, shuffled.length - 1);
    }
    if (shuffled.length === 1) {
        valCount = 1;
    }
    const valSet = new Set(shuffled.slice(0, valCount));
    if (valSet.size === 0 && shuffled.length) {
        valSet.add(shuffled[0]);
    }
    if (valSet.size === shuffled.length && shuffled.length > 1) {
        valSet.delete(shuffled[shuffled.length - 1]);
    }
    return valSet;
}

async function initQwenDatasetUpload(runName) {
    const formData = new FormData();
    if (runName) {
        formData.append("run_name", runName);
    }
    const resp = await fetch(`${API_ROOT}/qwen/dataset/init`, {
        method: "POST",
        body: formData,
    });
    if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || "Failed to initialize Qwen dataset upload.");
    }
    return resp.json();
}

async function uploadQwenDatasetChunk(jobId, split, record) {
    const formData = new FormData();
    formData.append("job_id", jobId);
    formData.append("split", split);
    formData.append("image_name", record.imageName);
    formData.append("annotation_line", record.annotation);
    formData.append("file", record.file, record.file?.name || record.imageName);
    const resp = await fetch(`${API_ROOT}/qwen/dataset/chunk`, {
        method: "POST",
        body: formData,
    });
    if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || `Failed to upload ${split} chunk (${resp.status})`);
    }
    return resp.json();
}

async function finalizeQwenDatasetUpload(jobId, metadata, runName) {
    const formData = new FormData();
    formData.append("job_id", jobId);
    formData.append("metadata", JSON.stringify(metadata || {}));
    if (runName) {
        formData.append("run_name", runName);
    }
    const resp = await fetch(`${API_ROOT}/qwen/dataset/finalize`, {
        method: "POST",
        body: formData,
    });
    if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || `Dataset finalize failed (${resp.status})`);
    }
    return resp.json();
}

async function cancelQwenDatasetUpload(jobId) {
    if (!jobId) {
        return;
    }
    try {
        const formData = new FormData();
        formData.append("job_id", jobId);
        await fetch(`${API_ROOT}/qwen/dataset/cancel`, {
            method: "POST",
            body: formData,
        });
    } catch (error) {
        console.debug("Failed to cancel Qwen dataset upload", error);
    }
}

async function uploadQwenDatasetStream() {
    const imageNames = Object.keys(images);
    if (!imageNames.length) {
        throw new Error("Load images before starting Qwen training.");
    }
    const classNames = Object.keys(classes || {});
    if (!classNames.length) {
        throw new Error("Load a label map before starting Qwen training.");
    }
    const contextText = qwenTrainElements.contextInput?.value?.trim() || "";
    const instruction = buildQwenInstruction(contextText, classNames);
    const stats = computeQwenDatasetStats(imageNames);
    let packagingModalVisible = false;
    let jobId = null;
    try {
        const summaryText = `${stats.imageCount} image${stats.imageCount === 1 ? "" : "s"} (${formatBytes(stats.imageBytes)}) + streaming annotations`;
        showTrainingPackagingModal(stats, {
            indeterminate: false,
            progressText: "Preparing dataset…",
            hintText: "Keep this tab open while we package the dataset for Qwen training.",
            summaryText,
        });
        packagingModalVisible = true;
        const runName = qwenTrainElements.runNameInput?.value?.trim() || "qwen_dataset";
        const initInfo = await initQwenDatasetUpload(runName);
        jobId = initInfo?.job_id;
        if (!jobId) {
            throw new Error("Dataset upload job id missing in response.");
        }
        const usedNames = new Map();
        const valSet = pickQwenValidationSet(imageNames);
        let processed = 0;
        let totalBoxes = 0;
        let uploadedTrain = 0;
        let uploadedVal = 0;
        let firstTrainRecord = null;
        let firstValRecord = null;
        for (const imageKey of imageNames) {
            const imageRecord = images[imageKey];
            if (!imageRecord || !imageRecord.meta) {
                throw new Error(`Missing original file for ${imageKey}. Re-import the images and try again.`);
            }
            await ensureImageDimensions(imageRecord);
            const baseName = imageRecord.meta.name || imageKey;
            const safeName = makeUniqueFilename(baseName, usedNames);
            const detections = buildDetectionRecords(imageKey, imageRecord);
            totalBoxes += detections.length;
            const annotation = JSON.stringify({
                image: safeName,
                context: instruction,
                detections,
            });
            const split = valSet.has(imageKey) ? "val" : "train";
            const recordPayload = {
                imageName: safeName,
                annotation,
                file: imageRecord.meta,
            };
            await uploadQwenDatasetChunk(jobId, split, recordPayload);
            if (split === "train") {
                uploadedTrain += 1;
                if (!firstTrainRecord) {
                    firstTrainRecord = recordPayload;
                }
            } else {
                uploadedVal += 1;
                if (!firstValRecord) {
                    firstValRecord = recordPayload;
                }
            }
            processed += 1;
            if (stats.imageCount > 0) {
                const percent = Math.min(95, Math.round((processed / stats.imageCount) * 90));
                updateTrainingPackagingProgress(percent, `Uploading dataset… ${percent}%`);
            }
        }
        if (uploadedTrain === 0 && firstValRecord) {
            await uploadQwenDatasetChunk(jobId, "train", firstValRecord);
            uploadedTrain += 1;
        }
        if (uploadedVal === 0 && firstTrainRecord) {
            await uploadQwenDatasetChunk(jobId, "val", firstTrainRecord);
            uploadedVal += 1;
        }
        if (totalBoxes === 0) {
            throw new Error("No bounding boxes available. Draw bboxes before training.");
        }
        const datasetMeta = {
            context: contextText,
            classes: classNames,
            created_at: Date.now(),
        };
        const finalizeInfo = await finalizeQwenDatasetUpload(jobId, datasetMeta, runName);
        updateTrainingPackagingProgress(100, "Dataset staged");
        return finalizeInfo;
    } catch (error) {
        await cancelQwenDatasetUpload(jobId);
        throw error;
    } finally {
        if (packagingModalVisible) {
            hideTrainingPackagingModal();
        }
    }
}

function buildQwenTrainingPayload(datasetRoot, datasetRunName) {
    const payload = { dataset_root: datasetRoot };
    const runName = qwenTrainElements.runNameInput?.value?.trim() || datasetRunName;
    if (runName) {
        payload.run_name = runName;
    }
    const modelId = qwenTrainElements.modelIdInput?.value?.trim();
    if (modelId) {
        payload.model_id = modelId;
    }
    const systemPrompt = qwenTrainElements.systemPromptInput?.value?.trim();
    if (systemPrompt) {
        payload.system_prompt = systemPrompt;
    }
    const promptNoise = readNumberInput(qwenTrainElements.systemPromptNoiseInput, { integer: false });
    if (promptNoise !== undefined) {
        payload.system_prompt_noise = promptNoise;
    }
    const accelerator = qwenTrainElements.acceleratorSelect?.value;
    if (accelerator) {
        payload.accelerator = accelerator;
    }
    payload.use_qlora = getSelectedQwenLoraMode() !== "lora";
    const numericMap = [
        ["batch_size", qwenTrainElements.batchSizeInput, { integer: true }],
        ["max_epochs", qwenTrainElements.epochsInput, { integer: true }],
        ["lr", qwenTrainElements.lrInput, { integer: false }],
        ["accumulate_grad_batches", qwenTrainElements.accumulateInput, { integer: true }],
        ["lora_rank", qwenTrainElements.loraRankInput, { integer: true }],
        ["lora_alpha", qwenTrainElements.loraAlphaInput, { integer: true }],
        ["lora_dropout", qwenTrainElements.loraDropoutInput, { integer: false }],
        ["patience", qwenTrainElements.patienceInput, { integer: true }],
    ];
    numericMap.forEach(([key, input, opts]) => {
        const value = readNumberInput(input, opts || {});
        if (value !== undefined) {
            payload[key] = value;
        }
    });
    return payload;
}

function setQwenSampleOverlay(text, variant) {
    const overlay = qwenTrainElements.sampleMessage;
    if (!overlay) {
        return;
    }
    overlay.textContent = text || "";
    overlay.classList.remove("hidden", "error", "success");
    if (variant) {
        overlay.classList.add(variant);
    }
    if (!text) {
        overlay.classList.add("hidden");
    }
}

function updateQwenSampleMeta(text) {
    if (qwenTrainElements.sampleMeta) {
        qwenTrainElements.sampleMeta.textContent = text || "";
    }
}

async function generateRandomQwenSample() {
    if (!qwenTrainElements.sampleButton) {
        return;
    }
    qwenTrainElements.sampleButton.disabled = true;
    try {
        setQwenSampleOverlay("Building sample…", "");
        const sample = await buildRandomQwenSampleData();
        await renderQwenSamplePreview(sample);
        const scope = describeQwenSampleScope(sample.mode, sample.labels);
        updateQwenSampleMeta(`Image: ${sample.imageLabel} • ${sample.useBBox ? "Bounding boxes" : "Points"} • ${scope}`);
        setQwenSampleOverlay("", "");
    } catch (error) {
        console.error(error);
        clearQwenSamplePreview();
        setQwenSampleOverlay(error.message || "Unable to build sample", "error");
        updateQwenSampleMeta("");
    } finally {
        qwenTrainElements.sampleButton.disabled = false;
    }
}

function describeQwenSampleScope(mode, labels) {
    if (!labels || !labels.length || mode === "all") {
        return "All classes";
    }
    if (mode === "single") {
        return `Only '${labels[0]}'`;
    }
    return `Subset: ${labels.join(", ")}`;
}

async function buildRandomQwenSampleData() {
    const imageKeys = Object.keys(images || {});
    if (!imageKeys.length) {
        throw new Error("Load images before generating a sample.");
    }
    const classNames = Object.keys(classes || {});
    if (!classNames.length) {
        throw new Error("Load a label map before generating a sample.");
    }
    const randomKey = imageKeys[Math.floor(Math.random() * imageKeys.length)];
    const imageRecord = images[randomKey];
    if (!imageRecord || !imageRecord.meta) {
        throw new Error("Selected image is missing its source file. Re-import and try again.");
    }
    await ensureImageDimensions(imageRecord);
    if (!imageRecord.object) {
        await loadImageObject(imageRecord);
    }
    const detections = buildDetectionRecords(randomKey, imageRecord);
    const contextText = qwenTrainElements.contextInput?.value?.trim() || "";
    const datasetInstruction = buildQwenInstruction(contextText, classNames);
    const { labels, mode } = chooseQwenSampleLabelSet(detections);
    const filtered = filterDetectionsForLabels(detections, labels, mode);
    const useBBox = Math.random() < 0.5;
    const prompt = buildQwenSampleUserPrompt(datasetInstruction, labels, mode, useBBox ? "bbox" : "point");
    const expectedDetections = filtered.map((det) => {
        const payload = { label: det.label };
        if (useBBox) {
            payload.bbox = det.bbox;
        } else {
            payload.point = det.point;
        }
        return payload;
    });
    return {
        imageKey: randomKey,
        imageLabel: imageRecord.meta.name || randomKey,
        imageRecord,
        prompt,
        expected: { detections: expectedDetections },
        labels,
        mode,
        useBBox,
    };
}

function clearQwenSamplePreview() {
    if (qwenTrainElements.sampleCanvas) {
        const ctx = qwenTrainElements.sampleCanvas.getContext("2d");
        if (ctx) {
            ctx.clearRect(0, 0, qwenTrainElements.sampleCanvas.width, qwenTrainElements.sampleCanvas.height);
        }
    }
    if (qwenTrainElements.samplePrompt) {
        qwenTrainElements.samplePrompt.textContent = "";
    }
    if (qwenTrainElements.sampleExpected) {
        qwenTrainElements.sampleExpected.textContent = "";
    }
}

async function renderQwenSamplePreview(sample) {
    if (!qwenTrainElements.sampleCanvas || !sample.imageRecord.object) {
        return;
    }
    const canvas = qwenTrainElements.sampleCanvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        return;
    }
    const width = sample.imageRecord.width || sample.imageRecord.object.width || 1;
    const height = sample.imageRecord.height || sample.imageRecord.object.height || 1;
    const maxW = 360;
    const maxH = 260;
    const scale = Math.min(maxW / width, maxH / height, 1);
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(sample.imageRecord.object, 0, 0, canvas.width, canvas.height);
    if (sample.useBBox) {
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = Math.max(1, 2 * scale);
        (sample.expected.detections || []).forEach((det) => {
            if (!det.bbox) {
                return;
            }
            const [x1, y1, x2, y2] = det.bbox;
            ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
        });
    } else {
        ctx.fillStyle = "#ef4444";
        const radius = Math.max(3, 4 * scale);
        (sample.expected.detections || []).forEach((det) => {
            if (!det.point) {
                return;
            }
            const [x, y] = det.point;
            ctx.beginPath();
            ctx.arc(x * scale, y * scale, radius, 0, Math.PI * 2);
            ctx.fill();
        });
    }
    if (qwenTrainElements.samplePrompt) {
        qwenTrainElements.samplePrompt.textContent = sample.prompt;
    }
    if (qwenTrainElements.sampleExpected) {
        qwenTrainElements.sampleExpected.textContent = JSON.stringify(sample.expected, null, 2);
    }
}

async function handleStartQwenTraining() {
    if (qwenTrainElements.startButton) {
        qwenTrainElements.startButton.disabled = true;
    }
    try {
        const datasetInfo = await uploadQwenDatasetStream();
        setQwenTrainMessage("Starting training job…");
        const payload = buildQwenTrainingPayload(datasetInfo.dataset_root, datasetInfo.run_name);
        const resp = await fetch(`${API_ROOT}/qwen/train/jobs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        qwenTrainState.activeJobId = data.job_id;
        setQwenTrainMessage("Job started", "success");
        if (qwenTrainElements.cancelButton) {
            qwenTrainElements.cancelButton.disabled = false;
        }
        await pollQwenTrainingJob(data.job_id, { force: true });
        await refreshQwenTrainingHistory();
    } catch (error) {
        console.error("Qwen training submit failed", error);
        setQwenTrainMessage(error.message || "Failed to start training", "error");
    } finally {
        if (qwenTrainElements.startButton) {
            qwenTrainElements.startButton.disabled = false;
        }
    }
}

async function cancelQwenTrainingJobRequest() {
    if (!qwenTrainState.activeJobId) {
        return;
    }
    try {
        const resp = await fetch(`${API_ROOT}/qwen/train/jobs/${qwenTrainState.activeJobId}/cancel`, { method: "POST" });
        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }
        setQwenTrainMessage("Cancellation requested", "warn");
    } catch (error) {
        console.error("Cancel Qwen job failed", error);
        setQwenTrainMessage(error.message || "Failed to cancel job", "error");
    }
}

function updateQwenTrainingUI(job) {
    if (!job) {
        return;
    }
    const pct = Math.round((job.progress || 0) * 100);
    if (qwenTrainElements.progressFill) {
        qwenTrainElements.progressFill.style.width = `${pct}%`;
    }
    if (qwenTrainElements.statusText) {
        const message = job.message ? ` • ${job.message}` : "";
        qwenTrainElements.statusText.textContent = `${job.status?.toUpperCase() || ""}${message}`;
    }
    if (qwenTrainElements.log) {
        const logs = Array.isArray(job.logs) ? job.logs : [];
        qwenTrainElements.log.textContent = logs
            .map((entry) => `[${formatTimestamp(entry.timestamp)}] ${entry.message}`)
            .join("\n");
    }
    if (qwenTrainElements.summary) {
        if (job.result) {
            const latest = job.result.latest || "–";
            const epochs = job.result.epochs_ran ?? "–";
            qwenTrainElements.summary.innerHTML = `
                <p><strong>Latest checkpoint:</strong> ${escapeHtml(latest)}</p>
                <p><strong>Epochs:</strong> ${escapeHtml(String(epochs))}</p>
            `;
        } else {
            qwenTrainElements.summary.textContent = "";
        }
    }
    if (job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") {
        if (qwenTrainElements.cancelButton) {
            qwenTrainElements.cancelButton.disabled = true;
        }
    }
}

function renderQwenTrainingHistoryItem(container, job) {
    if (!container) {
        return;
    }
    const item = document.createElement("div");
    item.className = "training-history-item";
    const label = job?.config?.run_name || job.job_id;
    const status = job.status || "unknown";
    const created = job.created_at ? new Date(job.created_at * 1000).toLocaleString() : "";
    const left = document.createElement("div");
    left.innerHTML = `<strong>${escapeHtml(label)}</strong><div class="training-help">${escapeHtml(status)} • ${escapeHtml(created)}</div>`;
    const right = document.createElement("div");
    const viewBtn = document.createElement("button");
    viewBtn.type = "button";
    viewBtn.className = "training-button";
    viewBtn.textContent = "View";
    viewBtn.addEventListener("click", () => {
        qwenTrainState.activeJobId = job.job_id;
        pollQwenTrainingJob(job.job_id, { force: true }).catch((error) => console.error("Poll Qwen job failed", error));
    });
    right.appendChild(viewBtn);
    item.append(left, right);
    container.appendChild(item);
}

async function refreshQwenTrainingHistory() {
    if (!qwenTrainElements.historyContainer) {
        return;
    }
    try {
        const resp = await fetch(`${API_ROOT}/qwen/train/jobs`);
        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }
        const jobs = await resp.json();
        qwenTrainElements.historyContainer.innerHTML = "";
        if (!Array.isArray(jobs) || !jobs.length) {
            const empty = document.createElement("div");
            empty.className = "training-history-item";
            empty.textContent = "No Qwen jobs yet.";
            qwenTrainElements.historyContainer.appendChild(empty);
            return;
        }
        jobs.forEach((job) => renderQwenTrainingHistoryItem(qwenTrainElements.historyContainer, job));
    } catch (error) {
        console.error("Failed to load Qwen job history", error);
        qwenTrainElements.historyContainer.textContent = `Unable to load history: ${error.message || error}`;
    }
}

function scheduleQwenJobPoll(jobId, delayMs = 1500) {
    if (qwenTrainState.pollHandle) {
        clearTimeout(qwenTrainState.pollHandle);
    }
    qwenTrainState.pollHandle = window.setTimeout(() => {
        pollQwenTrainingJob(jobId).catch((error) => console.error("Qwen poll failed", error));
    }, delayMs);
}

    async function pollQwenTrainingJob(jobId, { force = false } = {}) {
        if (!jobId) {
            return;
        }
        if (!force && activeTab !== TAB_QWEN_TRAIN) {
            return;
        }
        try {
            const resp = await fetch(`${API_ROOT}/qwen/train/jobs/${jobId}`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const job = await resp.json();
            qwenTrainState.activeJobId = job.job_id;
            updateQwenTrainingUI(job);
            if (job.status === "running" || job.status === "cancelling") {
                scheduleQwenJobPoll(job.job_id);
            } else if (qwenTrainState.pollHandle) {
                clearTimeout(qwenTrainState.pollHandle);
                qwenTrainState.pollHandle = null;
            }
        } catch (error) {
            console.error("pollQwenTrainingJob error", error);
            setQwenTrainMessage(error.message || "Unable to load job", "error");
        }
    }

    function initQwenTrainingTab() {
        if (qwenTrainElements.runNameInput) {
            return;
        }
        qwenTrainElements.runNameInput = document.getElementById("qwenTrainRunName");
        qwenTrainElements.contextInput = document.getElementById("qwenTrainContext");
        qwenTrainElements.modelIdInput = document.getElementById("qwenTrainModelId");
        qwenTrainElements.systemPromptInput = document.getElementById("qwenTrainSystemPrompt");
        qwenTrainElements.systemPromptNoiseInput = document.getElementById("qwenTrainPromptNoise");
        qwenTrainElements.acceleratorSelect = document.getElementById("qwenTrainAccelerator");
        qwenTrainElements.batchSizeInput = document.getElementById("qwenTrainBatchSize");
        qwenTrainElements.epochsInput = document.getElementById("qwenTrainEpochs");
        qwenTrainElements.lrInput = document.getElementById("qwenTrainLR");
        qwenTrainElements.accumulateInput = document.getElementById("qwenTrainAccumulate");
        qwenTrainElements.loraRankInput = document.getElementById("qwenTrainLoraRank");
        qwenTrainElements.loraAlphaInput = document.getElementById("qwenTrainLoraAlpha");
        qwenTrainElements.loraDropoutInput = document.getElementById("qwenTrainLoraDropout");
        qwenTrainElements.patienceInput = document.getElementById("qwenTrainPatience");
        qwenTrainElements.sampleButton = document.getElementById("qwenSampleBtn");
        qwenTrainElements.sampleCanvas = document.getElementById("qwenSampleCanvas");
        qwenTrainElements.samplePrompt = document.getElementById("qwenSamplePrompt");
        qwenTrainElements.sampleExpected = document.getElementById("qwenSampleExpected");
        qwenTrainElements.sampleMessage = document.getElementById("qwenSampleMessage");
        qwenTrainElements.sampleMeta = document.getElementById("qwenSampleMeta");
        qwenTrainElements.startButton = document.getElementById("qwenTrainStartBtn");
        qwenTrainElements.cancelButton = document.getElementById("qwenTrainCancelBtn");
        qwenTrainElements.progressFill = document.getElementById("qwenTrainProgressFill");
        qwenTrainElements.statusText = document.getElementById("qwenTrainStatusText");
        qwenTrainElements.message = document.getElementById("qwenTrainMessage");
        qwenTrainElements.summary = document.getElementById("qwenTrainSummary");
        qwenTrainElements.log = document.getElementById("qwenTrainLog");
        qwenTrainElements.historyContainer = document.getElementById("qwenTrainHistory");
        if (qwenTrainElements.startButton) {
            qwenTrainElements.startButton.addEventListener("click", () => {
                handleStartQwenTraining().catch((error) => console.error("Qwen training start failed", error));
            });
        }
        if (qwenTrainElements.cancelButton) {
            qwenTrainElements.cancelButton.addEventListener("click", () => {
                cancelQwenTrainingJobRequest().catch((error) => console.error("Qwen cancel failed", error));
            });
            qwenTrainElements.cancelButton.disabled = true;
        }
        if (qwenTrainElements.sampleButton) {
            qwenTrainElements.sampleButton.addEventListener("click", () => {
                generateRandomQwenSample().catch((error) => console.error("Random Qwen sample failed", error));
            });
        }
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

    const PACKAGING_REFERENCE_MBPS = 35;

    function computeDatasetStats(imageEntries, labelEntries) {
        const sumBytes = (entries) => entries.reduce((acc, entry) => acc + Math.max(0, entry.file?.size || 0), 0);
        const imageBytes = sumBytes(imageEntries);
        const labelBytes = sumBytes(labelEntries);
        const totalBytes = imageBytes + labelBytes;
        const totalFiles = imageEntries.length + labelEntries.length;
        const estimatedSeconds = totalBytes > 0
            ? totalBytes / (PACKAGING_REFERENCE_MBPS * 1024 * 1024)
            : null;
        return {
            imageCount: imageEntries.length,
            labelCount: labelEntries.length,
            totalFiles,
            imageBytes,
            labelBytes,
            totalBytes,
            estimatedSeconds,
        };
    }

    function replacePathLeaf(pathValue, replacement) {
        if (!pathValue) {
            return replacement;
        }
        const parts = pathValue.split(/[/\\]/);
        parts[parts.length - 1] = replacement;
        return parts.join("/");
    }

    function normaliseBaseName(entry) {
        if (!entry) {
            return null;
        }
        const source = entry.relativePath || entry.file?.name;
        if (!source) {
            return null;
        }
        const parts = source.split(/[/\\]/);
        return parts[parts.length - 1] || null;
    }

    function synthesiseLabelEntriesFromBboxes(imageEntries) {
        if (!Array.isArray(imageEntries) || !imageEntries.length) {
            return [];
        }
        const synthetic = [];
        imageEntries.forEach((entry) => {
            const baseName = normaliseBaseName(entry);
            if (!baseName) {
                return;
            }
            const bboxByClass = bboxes[baseName];
            const imageRecord = images[baseName];
            if (!bboxByClass || !imageRecord || !imageRecord.width || !imageRecord.height) {
                return;
            }
            const lines = [];
            Object.keys(bboxByClass).forEach((className) => {
                const classId = classes[className];
                if (classId === undefined || classId === null) {
                    return;
                }
                const records = bboxByClass[className] || [];
                records.forEach((bboxRecord) => {
                    const x = (bboxRecord.x + bboxRecord.width / 2) / imageRecord.width;
                    const y = (bboxRecord.y + bboxRecord.height / 2) / imageRecord.height;
                    const w = bboxRecord.width / imageRecord.width;
                    const h = bboxRecord.height / imageRecord.height;
                    lines.push(`${classId} ${x} ${y} ${w} ${h}`);
                });
            });
            if (!lines.length) {
                return;
            }
            const labelName = baseName.replace(/\.[^.]+$/, ".txt");
            const relativePath = entry.relativePath ? replacePathLeaf(entry.relativePath, labelName) : labelName;
            const blob = new Blob([lines.join("\n")], { type: "text/plain" });
            let syntheticFile;
            try {
                syntheticFile = new File([blob], labelName, { type: "text/plain" });
            } catch {
                syntheticFile = blob;
                syntheticFile.name = labelName;
            }
            synthetic.push({ file: syntheticFile, relativePath });
        });
        return synthetic;
    }

    function computeQwenDatasetStats(imageKeys) {
        let totalBytes = 0;
        imageKeys.forEach((key) => {
            const record = images[key];
            const file = record?.meta;
            if (file && typeof file.size === "number") {
                totalBytes += file.size;
            }
        });
        const estimatedSeconds = totalBytes > 0
            ? totalBytes / (PACKAGING_REFERENCE_MBPS * 1024 * 1024)
            : null;
        return {
            imageCount: imageKeys.length,
            labelCount: 0,
            totalFiles: imageKeys.length,
            imageBytes: totalBytes,
            labelBytes: 0,
            totalBytes,
            estimatedSeconds,
        };
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
        const usingUploads = !usingNativeImages;
        if (usingUploads) {
            imageEntries = getStoredEntries("images");
            if (!imageEntries.length) {
                throw new Error("Select an images folder that contains supported image files.");
            }
            labelEntries = getStoredEntries("labels");
        }
        let labelFallback = false;
        if (usingUploads && !labelEntries.length) {
            const synthetic = synthesiseLabelEntriesFromBboxes(imageEntries);
            if (synthetic.length) {
                labelEntries = synthetic;
                labelFallback = true;
            }
        }
        if (usingUploads && (!imageEntries.length || !labelEntries.length)) {
            throw new Error("Select an images folder and ensure label files or bounding boxes are available.");
        }
        const formData = new FormData();
        if (!usingUploads) {
            formData.append("images_path_native", trainingState.nativeImagesPath);
            formData.append("labels_path_native", trainingState.nativeLabelsPath);
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
        const datasetStats = usingUploads ? computeDatasetStats(imageEntries, labelEntries) : null;
        return { formData, usingUploads, imageEntries, labelEntries, datasetStats, labelFallback };
    }

    async function stageClipDatasetUploads(imageEntries, labelEntries) {
        if (!Array.isArray(imageEntries) || !imageEntries.length) {
            throw new Error("No images available for upload.");
        }
        if (!Array.isArray(labelEntries) || !labelEntries.length) {
            throw new Error("No label files found for upload.");
        }
        const initResp = await fetch(`${API_ROOT}/clip/dataset/init`, { method: "POST" });
        if (!initResp.ok) {
            const text = await initResp.text();
            throw new Error(text || "Failed to initialize dataset upload.");
        }
        const initData = await initResp.json();
        const jobId = initData?.job_id;
        if (!jobId) {
            throw new Error("Dataset upload job id missing.");
        }
        const totalItems = imageEntries.length + labelEntries.length;
        let completedItems = 0;

        const uploadEntry = async (entry, kind) => {
            const uploadForm = new FormData();
            uploadForm.append("job_id", jobId);
            uploadForm.append("kind", kind);
            const relPath = entry.relativePath || entry.file?.name || `${kind}_${completedItems}`;
            uploadForm.append("relative_path", relPath);
            uploadForm.append("file", entry.file, entry.file?.name || relPath);
            const resp = await fetch(`${API_ROOT}/clip/dataset/chunk`, {
                method: "POST",
                body: uploadForm,
            });
            if (!resp.ok) {
                const detail = await resp.text();
                throw new Error(detail || `Failed to upload ${kind}`);
            }
            completedItems += 1;
            if (totalItems > 0) {
                const percent = Math.min(100, Math.round((completedItems / totalItems) * 100));
                const stageLabel = kind === "image" ? "images" : "labels";
                updateTrainingPackagingProgress(percent, `Uploading ${stageLabel}… ${percent}%`);
            }
        };

        try {
            for (const entry of imageEntries) {
                await uploadEntry(entry, "image");
            }
            for (const entry of labelEntries) {
                await uploadEntry(entry, "label");
            }
            const finalizeForm = new FormData();
            finalizeForm.append("job_id", jobId);
            const finalizeResp = await fetch(`${API_ROOT}/clip/dataset/finalize`, {
                method: "POST",
                body: finalizeForm,
            });
            if (!finalizeResp.ok) {
                const detail = await finalizeResp.text();
                throw new Error(detail || "Failed to finalize dataset upload.");
            }
            return finalizeResp.json();
        } catch (error) {
            const cancelForm = new FormData();
            cancelForm.append("job_id", jobId);
            try {
                await fetch(`${API_ROOT}/clip/dataset/cancel`, { method: "POST", body: cancelForm });
            } catch (cancelError) {
                console.debug("Failed to cancel dataset upload job", cancelError);
            }
            throw error;
        }
    }

    async function handleStartTrainingClick() {
        if (!trainingElements.startButton) {
            return;
        }
        let packagingModalVisible = false;
        try {
            const { formData, usingUploads, imageEntries, labelEntries, datasetStats, labelFallback } = gatherTrainingFormData();
            trainingElements.startButton.disabled = true;
            const preppingMessage = usingUploads && datasetStats
                ? `Packaging dataset (${datasetStats.totalFiles} files ≈ ${formatBytes(datasetStats.totalBytes)}).`
                : "Submitting training job…";
            setTrainingMessage(preppingMessage, null);
            setActiveMessage(preppingMessage, null);
            if (usingUploads) {
                const statsForModal = datasetStats || {
                    imageCount: imageEntries.length,
                    labelCount: labelEntries.length,
                    totalFiles: imageEntries.length + labelEntries.length,
                    imageBytes: 0,
                    labelBytes: 0,
                    totalBytes: 0,
                };
                showTrainingPackagingModal(statsForModal, { indeterminate: false, progressText: "Uploading dataset…" });
                packagingModalVisible = true;
                try {
                    updateTrainingPackagingProgress(0, "Uploading images… 0%");
                    const stagingResult = await stageClipDatasetUploads(imageEntries, labelEntries);
                    formData.append("images_path_native", stagingResult.images_path);
                    formData.append("labels_path_native", stagingResult.labels_path);
                    if (stagingResult.temp_dir) {
                        formData.append("staged_temp_dir", stagingResult.temp_dir);
                    }
                } finally {
                    if (packagingModalVisible) {
                        hideTrainingPackagingModal();
                        packagingModalVisible = false;
                    }
                }
                if (labelFallback) {
                    setTrainingMessage("No label folder selected; using in-memory annotations.", "warn");
                }
            }
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
            if (packagingModalVisible) {
                hideTrainingPackagingModal();
            }
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
        tabElements.qwenTrainButton = document.getElementById("tabQwenTrainButton");
        tabElements.activeButton = document.getElementById("tabActiveButton");
        tabElements.qwenButton = document.getElementById("tabQwenButton");
        tabElements.predictorsButton = document.getElementById("tabPredictorsButton");
        tabElements.settingsButton = document.getElementById("tabSettingsButton");
        tabElements.labelingPanel = document.getElementById("tabLabeling");
        tabElements.trainingPanel = document.getElementById("tabTraining");
        tabElements.qwenTrainPanel = document.getElementById("tabQwenTrain");
        tabElements.activePanel = document.getElementById("tabActive");
        tabElements.qwenPanel = document.getElementById("tabQwen");
        tabElements.predictorsPanel = document.getElementById("tabPredictors");
        tabElements.settingsPanel = document.getElementById("tabSettings");
        if (tabElements.labelingButton) {
            tabElements.labelingButton.addEventListener("click", () => setActiveTab(TAB_LABELING));
        }
        if (tabElements.trainingButton) {
            tabElements.trainingButton.addEventListener("click", () => setActiveTab(TAB_TRAINING));
        }
        if (tabElements.qwenTrainButton) {
            tabElements.qwenTrainButton.addEventListener("click", () => setActiveTab(TAB_QWEN_TRAIN));
        }
        if (tabElements.activeButton) {
            tabElements.activeButton.addEventListener("click", () => setActiveTab(TAB_ACTIVE));
        }
        if (tabElements.qwenButton) {
            tabElements.qwenButton.addEventListener("click", () => setActiveTab(TAB_QWEN));
        }
        if (tabElements.predictorsButton) {
            tabElements.predictorsButton.addEventListener("click", () => setActiveTab(TAB_PREDICTORS));
        }
        if (tabElements.settingsButton) {
            tabElements.settingsButton.addEventListener("click", () => setActiveTab(TAB_SETTINGS));
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
        if (tabElements.qwenTrainButton) {
            tabElements.qwenTrainButton.classList.toggle("active", tabName === TAB_QWEN_TRAIN);
        }
        if (tabElements.activeButton) {
            tabElements.activeButton.classList.toggle("active", tabName === TAB_ACTIVE);
        }
        if (tabElements.qwenButton) {
            tabElements.qwenButton.classList.toggle("active", tabName === TAB_QWEN);
        }
        if (tabElements.predictorsButton) {
            tabElements.predictorsButton.classList.toggle("active", tabName === TAB_PREDICTORS);
        }
        if (tabElements.settingsButton) {
            tabElements.settingsButton.classList.toggle("active", tabName === TAB_SETTINGS);
        }
        if (tabElements.labelingPanel) {
            tabElements.labelingPanel.classList.toggle("active", tabName === TAB_LABELING);
        }
        if (tabElements.trainingPanel) {
            tabElements.trainingPanel.classList.toggle("active", tabName === TAB_TRAINING);
        }
        if (tabElements.qwenTrainPanel) {
            tabElements.qwenTrainPanel.classList.toggle("active", tabName === TAB_QWEN_TRAIN);
        }
        if (tabElements.activePanel) {
            tabElements.activePanel.classList.toggle("active", tabName === TAB_ACTIVE);
        }
        if (tabElements.qwenPanel) {
            tabElements.qwenPanel.classList.toggle("active", tabName === TAB_QWEN);
        }
        if (tabElements.predictorsPanel) {
            tabElements.predictorsPanel.classList.toggle("active", tabName === TAB_PREDICTORS);
        }
        if (tabElements.settingsPanel) {
            tabElements.settingsPanel.classList.toggle("active", tabName === TAB_SETTINGS);
        }
        if (tabName === TAB_TRAINING && previous !== TAB_TRAINING) {
            initializeTrainingUi();
            refreshTrainingHistory();
            populateClipBackbones();
            if (trainingState.activeJobId) {
                loadTrainingJob(trainingState.activeJobId, { forcePoll: true });
            }
        }
        if (tabName === TAB_QWEN_TRAIN && previous !== TAB_QWEN_TRAIN) {
            initQwenTrainingTab();
            refreshQwenTrainingHistory();
            if (qwenTrainState.activeJobId) {
                pollQwenTrainingJob(qwenTrainState.activeJobId, { force: true }).catch((error) => console.error("Qwen job poll failed", error));
            }
        }
        if (tabName === TAB_ACTIVE && previous !== TAB_ACTIVE) {
            initializeActiveModelUi();
            populateClipBackbones();
            refreshActiveModelPanel();
        }
        if (tabName === TAB_QWEN && previous !== TAB_QWEN) {
            initQwenModelTab();
        }
        if (tabName === TAB_PREDICTORS && previous !== TAB_PREDICTORS) {
            initializePredictorTab();
            startPredictorRefresh(true);
        } else if (previous === TAB_PREDICTORS && tabName !== TAB_PREDICTORS) {
            stopPredictorRefresh();
        }
        if (tabName === TAB_SETTINGS && !settingsUiInitialized) {
            initializeSettingsUi();
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
        const preserveExisting = uuid && tweakPreserveSet.has(uuid);
        if (preserveExisting) {
            tweakPreserveSet.delete(uuid);
        } else if (imageName && bboxes[imageName]) {
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

    function setBboxImportEnabled(enabled) {
        const bboxFileInput = document.getElementById("bboxes");
        const bboxFolderInput = document.getElementById("bboxesFolder");
        if (bboxFileInput) {
            bboxFileInput.disabled = !enabled;
        }
        if (bboxFolderInput) {
            bboxFolderInput.disabled = !enabled;
        }
        const bboxFileButton = document.getElementById("bboxesSelect");
        const bboxFolderButton = document.getElementById("bboxesSelectFolder");
        setButtonDisabled(bboxFileButton, !enabled);
        setButtonDisabled(bboxFolderButton, !enabled);
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

    function abortSlotPreload(slotName, options = {}) {
        const existing = slotPreloadControllers[slotName];
        if (!existing) {
            return;
        }
        const preserveSet = options?.preserveImages instanceof Set
            ? options.preserveImages
            : toImageNameSet(options?.preserveImages);
        if (preserveSet && existing.imageName && preserveSet.has(existing.imageName)) {
            return;
        }
        try {
            existing.controller?.abort();
        } catch (err) {
            console.debug(`Slot preload abort (${slotName}) failed`, err);
        }
        if (typeof existing.releaseLoading === "function") {
            existing.releaseLoading();
        }
        if (existing.imageName) {
            slotPreloadPromises.delete(existing.imageName);
            slotLoadingIndicators.delete(existing.imageName);
        }
        slotPreloadControllers[slotName] = null;
        scheduleSamSlotStatusRefresh(true);
    }

    function getNeighborSlots(currentName) {
        const listEl = document.getElementById("imageList");
        if (!listEl || !currentName) {
            return { nextName: null, previousName: null };
        }
        const options = Array.from(listEl.options);
        const idx = options.findIndex((opt) => getOptionImageName(opt) === currentName);
        if (idx === -1) {
            return { nextName: null, previousName: null };
        }
        const nextOpt = idx < options.length - 1 ? options[idx + 1] : null;
        const prevOpt = idx > 0 ? options[idx - 1] : null;
        return {
            nextName: getOptionImageName(nextOpt),
            previousName: getOptionImageName(prevOpt),
        };
    }

    function getOptionImageName(option) {
        if (!option) {
            return null;
        }
        return option.value || option.text || option.innerHTML || null;
    }

    function isSlotRoleEnabled(slotRole) {
        if (!slotRole || slotRole === "current") {
            return true;
        }
        if (slotRole === "next") {
            return samPredictorBudget >= 2;
        }
        if (slotRole === "previous") {
            return samPredictorBudget >= 3;
        }
        return true;
    }

    function ensureImageListVisibility(targetIndex) {
        const listEl = document.getElementById("imageList");
        if (!listEl || typeof targetIndex !== "number" || targetIndex < 0) {
            return;
        }
        const option = listEl.options[targetIndex];
        if (!option) {
            return;
        }
        const optionHeight = option.offsetHeight || parseInt(window.getComputedStyle(option).lineHeight || "0", 10) || 18;
        const listHeight = listEl.clientHeight || (optionHeight * listEl.size) || optionHeight;
        const visibleCount = optionHeight ? Math.max(1, Math.floor(listHeight / optionHeight)) : listEl.options.length;
        if (visibleCount >= listEl.options.length) {
            listEl.scrollTop = 0;
            return;
        }
        const halfWindow = Math.max(1, Math.floor(visibleCount / 2));
        const lastIndex = listEl.options.length - 1;
        const maxStart = Math.max(0, listEl.options.length - visibleCount);
        let firstIndex;
        if (targetIndex <= halfWindow) {
            firstIndex = 0;
        } else if (targetIndex >= lastIndex - halfWindow) {
            firstIndex = maxStart;
        } else {
            firstIndex = targetIndex - halfWindow;
        }
        const desiredScroll = firstIndex * optionHeight;
        if (Math.abs(listEl.scrollTop - desiredScroll) > 1) {
            listEl.scrollTop = desiredScroll;
        }
    }

    function syncImageSelectionToName(imageName, options = {}) {
        const listEl = document.getElementById("imageList");
        if (!listEl || !imageName) {
            return;
        }
        const opts = Array.from(listEl.options);
        const targetIndex = opts.findIndex((opt) => getOptionImageName(opt) === imageName);
        if (targetIndex === -1) {
            return;
        }
        const releaseLock = lockImageSelection();
        opts.forEach((opt, idx) => {
            opt.selected = idx === targetIndex;
        });
        listEl.selectedIndex = targetIndex;
        imageListIndex = targetIndex;
        if (options.ensureVisible) {
            ensureImageListVisibility(targetIndex);
        }
        releaseLock();
    }

    function lockImageSelection() {
        imageListSelectionLock++;
        let released = false;
        return () => {
            if (released) {
                return;
            }
            released = true;
            imageListSelectionLock = Math.max(0, imageListSelectionLock - 1);
        };
    }

    function toImageNameSet(values) {
        if (!values) {
            return null;
        }
        if (values instanceof Set) {
            return values;
        }
        if (!Array.isArray(values)) {
            return new Set(values ? [values] : []);
        }
        const filtered = values.filter(Boolean);
        return filtered.length ? new Set(filtered) : null;
    }

    function triggerNeighborSlotPreloads(currentName) {
        if (!samPreloadEnabled || !samSlotsEnabled || !currentName) {
            abortSlotPreload("next");
            abortSlotPreload("previous");
            return;
        }
        const { nextName, previousName } = getNeighborSlots(currentName);
        if (!isSlotRoleEnabled("next")) {
            abortSlotPreload("next");
        } else {
            if (nextName && shouldPreloadNeighborImage(nextName, "next")) {
                preloadSlotForImage(nextName, "next").catch((err) => {
                    console.debug("Next-slot preload failure", err);
                });
            } else if (!nextName) {
                abortSlotPreload("next");
            }
        }
        if (!isSlotRoleEnabled("previous")) {
            abortSlotPreload("previous");
        } else {
            if (previousName && shouldPreloadNeighborImage(previousName, "previous")) {
                preloadSlotForImage(previousName, "previous").catch((err) => {
                    console.debug("Previous-slot preload failure", err);
                });
            } else if (!previousName) {
                abortSlotPreload("previous");
            }
        }
    }

    async function activateImageSlot(imageName) {
        if (!samPreloadEnabled || !imageName) {
            return false;
        }
        if (!samSlotsEnabled) {
            const supported = await ensureSamSlotsSupport();
            if (!supported) {
                return false;
            }
        }
        const taskId = enqueueTask({ kind: "sam-activate", imageName });
        try {
            const resp = await fetch(`${API_ROOT}/sam_activate_slot`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_name: imageName, sam_variant: samVariant }),
            });
            if (!resp.ok) {
                completeTask(taskId);
                return false;
            }
            await resp.json();
            scheduleSamSlotStatusRefresh(true);
            completeTask(taskId);
            return true;
        } catch (error) {
            console.debug("activateImageSlot failed", error);
            completeTask(taskId);
            return false;
        }
    }

    async function prepareSamForCurrentImage(options = {}) {
        if (!samPreloadEnabled || !currentImage || !currentImage.name) {
            return;
        }
        const targetName = currentImage.name;
        const { messagePrefix = null, immediate = false } = options;
        const variantSnapshot = samVariant;
        const preloadAlreadyRunning = isSamPreloadActiveFor(targetName, variantSnapshot);
        if (samSlotsEnabled || await ensureSamSlotsSupport()) {
            let activated = await activateImageSlot(targetName);
            if (!activated && samSlotsEnabled) {
                const waited = await waitForSlotPreload(targetName);
                if (waited && currentImage && currentImage.name === targetName) {
                    activated = await activateImageSlot(targetName);
                }
            }
            if (activated) {
                if (!currentImage || currentImage.name !== targetName) {
                    return;
                }
                clearImageSlotLoading(targetName);
                setSamStatus(`SAM ready for ${targetName}`, { variant: "success", duration: 2000 });
                resolveSamPreloadWaiters(targetName, samVariant);
                hideSamPreloadProgress();
                triggerNeighborSlotPreloads(targetName);
                scheduleSamSlotStatusRefresh(true);
                return;
            }
        }
        scheduleSamPreload({
            force: !preloadAlreadyRunning,
            delayMs: immediate ? 0 : SAM_PRELOAD_IMAGE_SWITCH_DELAY_MS,
            messagePrefix,
            slot: samSlotsEnabled ? "current" : undefined,
            variant: variantSnapshot,
        });
    }

    async function ensureImageRecordReady(imageRecord) {
        if (!imageRecord) {
            return false;
        }
        if (imageRecord.object) {
            imageRecord.width = imageRecord.width || imageRecord.object.naturalWidth || imageRecord.object.width || imageRecord.width || 0;
            imageRecord.height = imageRecord.height || imageRecord.object.naturalHeight || imageRecord.object.height || imageRecord.height || 0;
            return true;
        }
        if (!imageRecord.meta) {
            return false;
        }
        try {
            await loadImageObject(imageRecord);
            imageRecord.width = imageRecord.width || imageRecord.object?.naturalWidth || imageRecord.object?.width || imageRecord.width || 0;
            imageRecord.height = imageRecord.height || imageRecord.object?.naturalHeight || imageRecord.object?.height || imageRecord.height || 0;
            return Boolean(imageRecord.object);
        } catch (err) {
            console.warn("Failed to load image record", imageRecord?.meta?.name, err);
            return false;
        }
    }

    async function getBase64ForImageRecord(imageRecord) {
        if (!imageRecord) {
            return null;
        }
        if (imageRecord.dataUrl && imageRecord.dataUrl.includes(',')) {
            return imageRecord.dataUrl.split(',')[1];
        }
        const ready = await ensureImageRecordReady(imageRecord);
        if (!ready || !imageRecord.object) {
            return null;
        }
        const offCanvas = document.createElement("canvas");
        offCanvas.width = imageRecord.width || imageRecord.object.naturalWidth || imageRecord.object.width || 0;
        offCanvas.height = imageRecord.height || imageRecord.object.naturalHeight || imageRecord.object.height || 0;
        if (!offCanvas.width || !offCanvas.height) {
            return null;
        }
        const ctx = offCanvas.getContext("2d");
        ctx.drawImage(imageRecord.object, 0, 0, offCanvas.width, offCanvas.height);
        const dataUrl = offCanvas.toDataURL("image/jpeg");
        imageRecord.dataUrl = dataUrl;
        return dataUrl.split(',')[1];
    }

    async function preloadSlotForImage(imageName, slotName) {
        if (!samPreloadEnabled || !imageName) {
            return null;
        }
        if (!samSlotsEnabled) {
            const supported = await ensureSamSlotsSupport();
            if (!supported) {
                return null;
            }
        }
        const imageRecord = images[imageName];
        if (!imageRecord) {
            return null;
        }
        const existingPromise = slotPreloadPromises.get(imageName);
        if (existingPromise) {
            return existingPromise;
        }
        const existingTask = slotPreloadControllers[slotName];
        if (existingTask && existingTask.imageName === imageName) {
            const promise = slotPreloadPromises.get(imageName);
            if (promise) {
                return promise;
            }
        }
        abortSlotPreload(slotName);

        const runPromise = (async () => {
            const controller = new AbortController();
            const slotTask = { controller, releaseLoading: null, imageName };
            slotPreloadControllers[slotName] = slotTask;
            try {
                const requestBody = { slot: slotName, sam_variant: samVariant, image_name: imageName };
                const cachedToken = getSamToken(imageName, samVariant);
                if (cachedToken) {
                    requestBody.image_token = cachedToken;
                } else {
                    const base64Img = await getBase64ForImageRecord(imageRecord);
                    if (!base64Img) {
                        if (slotPreloadControllers[slotName] === slotTask) {
                            if (typeof slotTask.releaseLoading === "function") {
                                slotTask.releaseLoading();
                            }
                            slotPreloadControllers[slotName] = null;
                        }
                        return null;
                    }
                    requestBody.image_base64 = base64Img;
                }

                slotTask.releaseLoading = beginImageSlotLoading(imageName, slotName);
                const resp = await fetch(`${API_ROOT}/sam_preload`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(requestBody),
                    signal: controller.signal,
                });
                if (!resp.ok) {
                    const detail = await resp.text();
                    throw new Error(detail || `HTTP ${resp.status}`);
                }
                const result = await resp.json();
                if (result?.token) {
                    rememberSamToken(imageName, samVariant, result.token);
                }
                return result;
            } catch (error) {
                if (!error || error.name !== "AbortError") {
                    console.warn(`Background SAM preload failed for ${imageName} (${slotName})`, error);
                }
                throw error;
            } finally {
                const activeTask = slotPreloadControllers[slotName];
                if (activeTask === slotTask) {
                    if (typeof slotTask.releaseLoading === "function") {
                        slotTask.releaseLoading();
                    }
                    slotPreloadControllers[slotName] = null;
                }
                slotPreloadPromises.delete(imageName);
                scheduleSamSlotStatusRefresh();
            }
        })();

        slotPreloadPromises.set(imageName, runPromise);
        runPromise.catch(() => null);
        return runPromise;
    }

    async function ensureSamSlotsSupport() {
        if (samSlotsSupportChecked) {
            return samSlotsEnabled;
        }
        samSlotsSupportChecked = true;
        try {
            const resp = await fetch(`${API_ROOT}/sam_slots`);
            if (!resp.ok) {
                throw new Error(await resp.text());
            }
            const data = await resp.json();
            samSlotsEnabled = Array.isArray(data);
            if (samSlotsEnabled) {
                updateSlotHighlights(Array.isArray(data) ? data : []);
            }
        } catch (error) {
            samSlotsEnabled = false;
        }
        return samSlotsEnabled;
    }

    function scheduleSamSlotStatusRefresh(immediate = false) {
        if (!samSlotsEnabled || !samPreloadEnabled) {
            if (samSlotStatusTimer) {
                clearTimeout(samSlotStatusTimer);
                samSlotStatusTimer = null;
            }
            updateSlotHighlights([]);
            return;
        }
        if (samSlotStatusTimer) {
            clearTimeout(samSlotStatusTimer);
            samSlotStatusTimer = null;
        }
        if (immediate) {
            if (samSlotStatusPending) {
                samSlotStatusNeedsRefresh = true;
            } else {
                refreshSamSlotStatus();
            }
            return;
        }
        samSlotStatusTimer = setTimeout(() => {
            samSlotStatusTimer = null;
            if (samSlotStatusPending) {
                samSlotStatusNeedsRefresh = true;
            } else {
                refreshSamSlotStatus();
            }
        }, SAM_SLOT_STATUS_DEBOUNCE_MS);
    }

    async function refreshSamSlotStatus() {
        if (!samSlotsEnabled || !samPreloadEnabled) {
            return;
        }
        if (samSlotStatusPending) {
            samSlotStatusNeedsRefresh = true;
            return;
        }
        samSlotStatusPending = true;
        try {
            const resp = await fetch(`${API_ROOT}/sam_slots`);
            if (!resp.ok) {
                if (resp.status === 404) {
                    samSlotsEnabled = false;
                    updateSlotHighlights([]);
                    return;
                }
                throw new Error(await resp.text());
            }
            const data = await resp.json();
            updateSlotHighlights(Array.isArray(data) ? data : []);
        } catch (error) {
            console.debug("SAM slot status refresh failed", error);
        } finally {
            samSlotStatusPending = false;
            if (samSlotStatusNeedsRefresh) {
                samSlotStatusNeedsRefresh = false;
                refreshSamSlotStatus();
            }
        }
    }

    function beginImageSlotLoading(imageName, slotName = "current") {
        if (!imageName) {
            return () => {};
        }
        const normalizedSlot = slotName || "current";
        let entry = slotLoadingIndicators.get(imageName);
        if (!entry) {
            entry = { slots: new Map() };
            slotLoadingIndicators.set(imageName, entry);
        }
        const currentCount = entry.slots.get(normalizedSlot) || 0;
        entry.slots.set(normalizedSlot, currentCount + 1);
        applySlotStatusClasses();
        let released = false;
        return () => {
            if (released) {
                return;
            }
            released = true;
            const activeEntry = slotLoadingIndicators.get(imageName);
            if (!activeEntry) {
                applySlotStatusClasses();
                return;
            }
            const remaining = (activeEntry.slots.get(normalizedSlot) || 1) - 1;
            if (remaining <= 0) {
                activeEntry.slots.delete(normalizedSlot);
            } else {
                activeEntry.slots.set(normalizedSlot, remaining);
            }
            if (activeEntry.slots.size === 0) {
                slotLoadingIndicators.delete(imageName);
            }
            applySlotStatusClasses();
        };
    }

    function clearImageSlotLoading(imageName) {
        if (!imageName) {
            return;
        }
        if (slotLoadingIndicators.delete(imageName)) {
            applySlotStatusClasses();
        }
    }

    function applySlotStatusClasses(options = {}) {
        const imageList = document.getElementById("imageList");
        if (!imageList) {
            return;
        }
        const slotStatusMap = new Map();
        latestSlotStatuses.forEach((entry) => {
            if (entry?.image_name) {
                slotStatusMap.set(entry.image_name, entry);
            }
        });
        const optionsArray = Array.from(imageList.options);
        let needsSelectionFix = false;
        optionsArray.forEach((option) => {
            option.classList.remove(
                "sam-slot-current",
                "sam-slot-next",
                "sam-slot-previous",
                "sam-slot-loaded",
                "sam-slot-loading"
            );
            const name = option.value || option.innerHTML;
            const slotEntry = slotStatusMap.get(name);
            const loadingEntry = slotLoadingIndicators.get(name);
            if (slotEntry?.slot) {
                option.classList.add(`sam-slot-${slotEntry.slot}`);
            }
            if (loadingEntry?.slots?.size) {
                loadingEntry.slots.forEach((count, slotName) => {
                    if (slotName && count > 0) {
                        option.classList.add(`sam-slot-${slotName}`);
                    }
                });
                option.classList.add("sam-slot-loading");
            } else if (slotEntry?.slot) {
                option.classList.add("sam-slot-loaded");
                if (slotEntry.busy) {
                    option.classList.add("sam-slot-loading");
                }
            }
            if (
                !needsSelectionFix &&
                currentImage &&
                currentImage.name === name &&
                !option.selected
            ) {
                needsSelectionFix = true;
            }
        });
        if (needsSelectionFix && imageListSelectionLock === 0) {
            syncImageSelectionToName(currentImage.name, { ensureVisible: false });
        }
    }

    function updateSlotHighlights(statusList) {
        latestSlotStatuses = Array.isArray(statusList) ? statusList : [];
        applySlotStatusClasses();
    }

    function findSlotStatusForImage(imageName, variant = samVariant) {
        if (!imageName) {
            return null;
        }
        return latestSlotStatuses.find((entry) => {
            if (!entry || entry.image_name !== imageName) {
                return false;
            }
            if (!entry.variant || !variant) {
                return true;
            }
            return entry.variant === variant;
        }) || null;
    }

    function isImageCurrentlyLoading(imageName) {
        if (!imageName) {
            return false;
        }
        return slotLoadingIndicators.has(imageName);
    }

    function shouldPreloadNeighborImage(imageName, slotRole = "current", variant = samVariant) {
        if (!imageName) {
            return false;
        }
        if (currentImage && currentImage.name === imageName) {
            return false;
        }
        if (!isSlotRoleEnabled(slotRole)) {
            return false;
        }
        if (isImageCurrentlyLoading(imageName)) {
            return false;
        }
        const entry = findSlotStatusForImage(imageName, variant);
        return !entry;
    }

    function waitForSlotPreload(imageName) {
        if (!imageName) {
            return Promise.resolve(false);
        }
        const promise = slotPreloadPromises.get(imageName);
        if (!promise) {
            return Promise.resolve(false);
        }
        return promise.then(() => true).catch(() => false);
    }

    function setPredictorMessage(text, variant = "info") {
        if (!predictorElements.message) {
            return;
        }
        predictorElements.message.textContent = text || "";
        predictorElements.message.className = `predictor-message ${variant}`;
    }

    function coerceNumber(value, fallback) {
        if (value === null || typeof value === "undefined") {
            return fallback;
        }
        const parsed = Number(value);
        return Number.isNaN(parsed) ? fallback : parsed;
    }

    function applyPredictorState(data) {
        if (!data) {
            return;
        }
        predictorSettings = {
            maxPredictors: coerceNumber(data.max_predictors, predictorSettings.maxPredictors),
            minPredictors: coerceNumber(data.min_predictors, predictorSettings.minPredictors),
            maxSupportedPredictors: coerceNumber(data.max_supported_predictors, predictorSettings.maxSupportedPredictors),
            activePredictors: coerceNumber(data.active_predictors, predictorSettings.activePredictors),
            loadedPredictors: coerceNumber(data.loaded_predictors, predictorSettings.loadedPredictors),
            processRamMb: coerceNumber(data.process_ram_mb, 0),
            totalRamMb: coerceNumber(data.total_ram_mb, 0),
            availableRamMb: coerceNumber(data.available_ram_mb, 0),
            imageRamMb: coerceNumber(data.image_ram_mb, 0),
        };
        samPredictorBudget = predictorSettings.maxPredictors;
        if (predictorElements.countInput) {
            predictorElements.countInput.min = predictorSettings.minPredictors || 1;
            predictorElements.countInput.max = predictorSettings.maxSupportedPredictors || predictorElements.countInput.max || 3;
            predictorElements.countInput.value = predictorSettings.maxPredictors;
        }
        renderPredictorStats();
    }

    function renderPredictorStats() {
        if (predictorElements.activeCount) {
            predictorElements.activeCount.textContent = `${predictorSettings.activePredictors} / ${predictorSettings.maxSupportedPredictors}`;
        }
        if (predictorElements.loadedCount) {
            predictorElements.loadedCount.textContent = `${predictorSettings.loadedPredictors}`;
        }
        if (predictorElements.processRam) {
            predictorElements.processRam.textContent = formatMb(predictorSettings.processRamMb);
        }
        if (predictorElements.imageRam) {
            predictorElements.imageRam.textContent = formatMb(predictorSettings.imageRamMb);
        }
        if (predictorElements.systemFreeRam) {
            const free = predictorSettings.availableRamMb;
            const total = predictorSettings.totalRamMb;
            predictorElements.systemFreeRam.textContent = total
                ? `${formatMb(free)} / ${formatMb(total)}`
                : formatMb(free);
        }
    }

    function formatMb(value) {
        if (typeof value !== "number" || Number.isNaN(value)) {
            return "--";
        }
        return `${value.toFixed(1)} MB`;
    }

    async function refreshPredictorMetrics(options = {}) {
        if (predictorRefreshInFlight) {
            return;
        }
        predictorRefreshInFlight = true;
        try {
            const resp = await fetch(`${API_ROOT}/predictor_settings`);
            if (!resp.ok) {
                throw new Error(await resp.text() || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            applyPredictorState(data);
            if (!options.silent) {
                setPredictorMessage("Predictor stats updated.", "success");
            }
        } catch (error) {
            if (!options.silent) {
                setPredictorMessage(`Unable to fetch predictor stats: ${error.message || error}`, "error");
            }
        } finally {
            predictorRefreshInFlight = false;
        }
    }

    async function submitPredictorSettings(desiredCount) {
        const min = predictorSettings.minPredictors || 1;
        const max = predictorSettings.maxSupportedPredictors || 3;
        const normalized = Math.max(min, Math.min(max, Number(desiredCount) || min));
        setPredictorMessage("Updating predictor budget…", "info");
        try {
            const resp = await fetch(`${API_ROOT}/predictor_settings`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ max_predictors: normalized }),
            });
            if (!resp.ok) {
                throw new Error(await resp.text() || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            applyPredictorState(data);
            setPredictorMessage("Predictor budget updated.", "success");
        } catch (error) {
            setPredictorMessage(`Failed to update predictor budget: ${error.message || error}`, "error");
        }
    }

    function startPredictorRefresh(immediate = false) {
        if (predictorRefreshTimer) {
            clearInterval(predictorRefreshTimer);
            predictorRefreshTimer = null;
        }
        if (immediate) {
            refreshPredictorMetrics({ silent: true });
        }
        predictorRefreshTimer = setInterval(() => {
            refreshPredictorMetrics({ silent: true });
        }, PREDICTOR_REFRESH_INTERVAL_MS);
    }

    function stopPredictorRefresh() {
        if (predictorRefreshTimer) {
            clearInterval(predictorRefreshTimer);
            predictorRefreshTimer = null;
        }
    }

    function initializePredictorTab() {
        if (predictorTabInitialized) {
            return;
        }
        predictorTabInitialized = true;
        refreshPredictorMetrics({ silent: true });
    }

    function initializeSettingsUi() {
        if (settingsUiInitialized) {
            return;
        }
        settingsUiInitialized = true;
        settingsElements.apiInput = document.getElementById("settingsApiRoot");
        settingsElements.applyButton = document.getElementById("settingsApply");
        settingsElements.testButton = document.getElementById("settingsTest");
        settingsElements.status = document.getElementById("settingsStatus");
        if (settingsElements.apiInput) {
            settingsElements.apiInput.value = API_ROOT;
        }
        setSettingsStatus(`Current backend: ${API_ROOT}`, "info");
        if (settingsElements.applyButton) {
            settingsElements.applyButton.addEventListener("click", () => applyApiRootValue(settingsElements.apiInput?.value || ""));
        }
        if (settingsElements.testButton) {
            settingsElements.testButton.addEventListener("click", () => testApiRootCandidate(settingsElements.apiInput?.value || API_ROOT));
        }
    }

    function setSettingsStatus(message, variant = "info") {
        if (!settingsElements.status) {
            return;
        }
        settingsElements.status.textContent = message || "";
        settingsElements.status.className = variant ? `settings-status ${variant}` : "settings-status";
    }

    function initQwenPanel() {
        qwenElements.statusLabel = document.getElementById("qwenStatusLabel");
        qwenElements.itemsInput = document.getElementById("qwenItems");
        qwenElements.manualPrompt = document.getElementById("qwenCustomPrompt");
        qwenElements.imageTypeInput = document.getElementById("qwenImageType");
        qwenElements.extraContextInput = document.getElementById("qwenExtraContext");
        qwenElements.advancedToggle = document.getElementById("qwenAdvancedToggle");
        qwenElements.advancedPanel = document.getElementById("qwenAdvancedPanel");
        qwenElements.classSelect = document.getElementById("qwenClassSelect");
        qwenElements.promptType = document.getElementById("qwenPromptType");
        qwenElements.maxResults = document.getElementById("qwenMaxResults");
        qwenElements.runButton = document.getElementById("qwenRunButton");
        if (qwenElements.runButton) {
            qwenElements.runButton.addEventListener("click", () => {
                handleQwenRun().catch((error) => {
                    console.error("Qwen request failed", error);
                });
            });
        }
        if (qwenElements.classSelect) {
            qwenElements.classSelect.addEventListener("change", () => {
                qwenClassOverride = true;
            });
        }
        if (qwenElements.advancedToggle) {
            qwenElements.advancedToggle.addEventListener("click", () => toggleQwenAdvanced());
        }
        toggleQwenAdvanced(false);
        applyActiveQwenMetadata(qwenModelState.activeMetadata);
        updateQwenRunButton();
        updateQwenClassOptions({ resetOverride: true });
        refreshQwenStatus({ silent: true }).catch((error) => {
            console.debug("Unable to query Qwen status", error);
        });
    }

    function setQwenStatusLabel(message, state = "info") {
        if (!qwenElements.statusLabel) {
            return;
        }
        qwenElements.statusLabel.textContent = message;
        qwenElements.statusLabel.classList.remove("qwen-status-label--ready", "qwen-status-label--error");
        if (state === "ready") {
            qwenElements.statusLabel.classList.add("qwen-status-label--ready");
        } else if (state === "error") {
            qwenElements.statusLabel.classList.add("qwen-status-label--error");
        }
    }

    function orderedClassNames() {
        const entries = Object.entries(classes || {});
        if (!entries.length) {
            return [];
        }
        return entries
            .sort((a, b) => a[1] - b[1])
            .map(([name]) => name);
    }

    function updateQwenClassOptions({ resetOverride = false, preserveSelection = false } = {}) {
        if (!qwenElements.classSelect) {
            return;
        }
        const classNames = orderedClassNames();
        qwenElements.classSelect.innerHTML = "";
        if (!classNames.length) {
            const placeholder = document.createElement("option");
            placeholder.textContent = "Load classes first";
            placeholder.value = "";
            qwenElements.classSelect.appendChild(placeholder);
            qwenElements.classSelect.disabled = true;
            qwenClassOverride = false;
            return;
        }
        qwenElements.classSelect.disabled = false;
        const previousValue = preserveSelection ? qwenElements.classSelect.value : null;
        classNames.forEach((name) => {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            qwenElements.classSelect.appendChild(option);
        });
        let targetValue = null;
        if (preserveSelection && previousValue && classNames.includes(previousValue)) {
            targetValue = previousValue;
        } else if (currentClass && classNames.includes(currentClass)) {
            targetValue = currentClass;
        } else {
            targetValue = classNames[0];
        }
        qwenElements.classSelect.value = targetValue;
        if (resetOverride) {
            qwenClassOverride = false;
        }
    }

    function syncQwenClassToCurrent() {
        if (!qwenElements.classSelect || qwenClassOverride) {
            return;
        }
        if (currentClass && qwenElements.classSelect.value !== currentClass) {
            const options = Array.from(qwenElements.classSelect.options).map((opt) => opt.value);
            if (options.includes(currentClass)) {
                qwenElements.classSelect.value = currentClass;
            }
        }
    }

    function getQwenTargetClass() {
        if (qwenElements.classSelect && qwenElements.classSelect.value) {
            return qwenElements.classSelect.value;
        }
        return currentClass;
    }

    function updateQwenRunButton() {
        if (!qwenElements.runButton) {
            return;
        }
        qwenElements.runButton.disabled = !qwenAvailable || qwenRequestActive;
        qwenElements.runButton.textContent = qwenRequestActive ? "Running…" : "Use Qwen";
    }

    function toggleQwenAdvanced(forceState = null) {
        if (!qwenElements.advancedToggle || !qwenElements.advancedPanel) {
            return;
        }
        if (typeof forceState === "boolean") {
            qwenAdvancedVisible = forceState;
        } else {
            qwenAdvancedVisible = !qwenAdvancedVisible;
        }
        const expanded = qwenAdvancedVisible;
        qwenElements.advancedToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
        qwenElements.advancedPanel.setAttribute("aria-hidden", expanded ? "false" : "true");
        qwenElements.advancedToggle.textContent = expanded ? "Hide advanced overrides" : "Show advanced overrides";
    }

    async function refreshQwenStatus({ silent = false } = {}) {
        if (!qwenElements.statusLabel) {
            return;
        }
        if (!silent) {
            setQwenStatusLabel("Checking…", "info");
        }
        qwenAvailable = false;
        updateQwenRunButton();
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 6000);
            const resp = await fetch(`${API_ROOT}/qwen/status`, { signal: controller.signal });
            clearTimeout(timeout);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const data = await resp.json();
            if (data.available && !data.dependency_error) {
                qwenAvailable = true;
                const deviceLabel = data.device ? ` (${data.device})` : "";
                if (data.loaded) {
                    setQwenStatusLabel(`Ready${deviceLabel}`, "ready");
                } else {
                    setQwenStatusLabel(`Available${deviceLabel}`, "ready");
                }
            } else if (data.dependency_error) {
                setQwenStatusLabel("Unavailable (deps)", "error");
            } else {
                setQwenStatusLabel("Disabled", "error");
            }
        } catch (error) {
            setQwenStatusLabel("Unavailable", "error");
            qwenAvailable = false;
        } finally {
            updateQwenRunButton();
        }
    }

    function setQwenModelStatus(message, variant = "info") {
        if (!qwenModelElements.status) {
            return;
        }
        qwenModelElements.status.textContent = message || "";
        qwenModelElements.status.className = variant ? `qwen-model-status ${variant}` : "qwen-model-status";
    }

    function applyActiveQwenMetadata(metadata) {
        qwenModelState.activeMetadata = metadata || null;
        const badge = document.getElementById("qwenActiveModelLabel");
        if (badge) {
            badge.textContent = metadata?.label ? `Active: ${metadata.label}` : "Active: Base Qwen";
        }
        const context = metadata?.dataset_context || "";
        const classes = Array.isArray(metadata?.classes) ? metadata.classes : [];
        if (qwenElements.imageTypeInput && !qwenElements.imageTypeInput.value) {
            qwenElements.imageTypeInput.placeholder = context || "Describe the image";
        }
        if (qwenElements.itemsInput && !qwenElements.itemsInput.value) {
            qwenElements.itemsInput.placeholder = classes.length ? classes.join(", ") : "car, bus, kiosk";
        }
    }

    function renderQwenModelDetails(metadata) {
        if (!qwenModelElements.details) {
            return;
        }
        if (!metadata) {
            qwenModelElements.details.innerHTML = "Select a model to see its prompts and defaults.";
            return;
        }
        const classes = Array.isArray(metadata.classes) ? metadata.classes.join(", ") : "(not specified)";
        const context = metadata.dataset_context || "(not specified)";
        qwenModelElements.details.innerHTML = `
            <p><strong>Name:</strong> ${metadata.label || metadata.id || "Custom Run"}</p>
            <p><strong>Base model:</strong> ${metadata.model_id || "Qwen/Qwen2.5-VL-3B-Instruct"}</p>
            <p><strong>Context hint:</strong> ${context}</p>
            <p><strong>Classes:</strong> ${classes}</p>
            <label>System prompt</label>
            <pre>${metadata.system_prompt || ""}</pre>
        `;
    }

    function renderQwenModelList() {
        if (!qwenModelElements.list) {
            return;
        }
        qwenModelElements.list.innerHTML = "";
        qwenModelState.models.forEach((entry) => {
            const card = document.createElement("div");
            card.className = entry.active ? "qwen-model-card active" : "qwen-model-card";
            const title = document.createElement("h3");
            title.textContent = entry.label || entry.id;
            card.appendChild(title);
            const metaText = document.createElement("p");
            const context = entry.metadata?.dataset_context;
            const classes = Array.isArray(entry.metadata?.classes) ? entry.metadata.classes.join(", ") : "";
            metaText.textContent = [context, classes].filter(Boolean).join(" • ") || "No context provided";
            card.appendChild(metaText);
            const button = document.createElement("button");
            button.type = "button";
            button.className = "training-button";
            button.textContent = entry.active ? "Active" : "Activate";
            button.disabled = !!entry.active;
            button.addEventListener("click", () => activateQwenModel(entry.id));
            card.appendChild(button);
            qwenModelElements.list.appendChild(card);
        });
    }

    async function refreshQwenModels() {
        setQwenModelStatus("Loading models…", "info");
        try {
            const resp = await fetch(`${API_ROOT}/qwen/models`);
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            qwenModelState.models = data.models || [];
            qwenModelState.activeId = data.active || "default";
            const activeEntry = qwenModelState.models.find((entry) => entry.id === qwenModelState.activeId);
            applyActiveQwenMetadata(activeEntry?.metadata || null);
            renderQwenModelList();
            renderQwenModelDetails(activeEntry?.metadata || null);
            setQwenModelStatus("Models loaded.", "success");
        } catch (error) {
            console.error("Failed to load Qwen models", error);
            setQwenModelStatus(`Failed to load models: ${error.message || error}`, "error");
        }
    }

    async function activateQwenModel(modelId) {
        setQwenModelStatus("Switching models…", "info");
        try {
            const resp = await fetch(`${API_ROOT}/qwen/models/activate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_id: modelId }),
            });
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
            }
            await refreshQwenModels();
            setQwenModelStatus("Model activated.", "success");
        } catch (error) {
            console.error("Failed to activate Qwen model", error);
            setQwenModelStatus(`Activation failed: ${error.message || error}`, "error");
        }
    }

    function initQwenModelTab() {
        if (qwenModelElements.status) {
            return;
        }
        qwenModelElements.status = document.getElementById("qwenModelStatus");
        qwenModelElements.list = document.getElementById("qwenModelList");
        qwenModelElements.details = document.getElementById("qwenModelDetails");
        refreshQwenModels();
    }

    async function handleQwenRun() {
        if (qwenRequestActive) {
            return;
        }
        if (!qwenAvailable) {
            setSamStatus("Qwen backend is unavailable", { variant: "warn", duration: 3500 });
            return;
        }
        if (!currentImage || !currentImage.name) {
            setSamStatus("Load an image before using Qwen", { variant: "warn", duration: 3000 });
            return;
        }
        const targetClass = getQwenTargetClass();
        if (!targetClass) {
            setSamStatus("Load classes and pick a target class before using Qwen", { variant: "warn", duration: 3500 });
            return;
        }
        const manualPrompt = (qwenElements.manualPrompt?.value || "").trim();
        const itemsText = (qwenElements.itemsInput?.value || "").trim();
        if (!manualPrompt && !itemsText) {
            setSamStatus("Describe what to detect or supply a custom prompt.", { variant: "warn", duration: 3500 });
            qwenElements.itemsInput?.focus();
            return;
        }
        const promptType = qwenElements.promptType?.value || "bbox";
        let maxResults = parseInt(qwenElements.maxResults?.value || "8", 10);
        if (Number.isNaN(maxResults)) {
            maxResults = 8;
        }
        maxResults = Math.min(Math.max(maxResults, 1), 50);
        qwenRequestActive = true;
        updateQwenRunButton();
        setSamStatus(`Running Qwen (${promptType === "point" ? "points" : "bbox"}) for ${targetClass}…`, { variant: "info", duration: 0 });
        try {
            const requestFields = {
                prompt_type: promptType,
                max_results: maxResults,
            };
            if (manualPrompt) {
                requestFields.prompt = manualPrompt;
            } else {
                requestFields.item_list = itemsText;
                const imageTypeOverride = (qwenElements.imageTypeInput?.value || "").trim();
                const extraOverride = (qwenElements.extraContextInput?.value || "").trim();
                if (imageTypeOverride) {
                    requestFields.image_type = imageTypeOverride;
                }
                if (extraOverride) {
                    requestFields.extra_context = extraOverride;
                }
            }
            const result = await invokeQwenInfer(requestFields);
            if (currentImage && result?.image_token) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            const added = applyQwenBoxes(result?.boxes || [], targetClass);
            if (!added) {
                const warning = Array.isArray(result?.warnings) && result.warnings.includes("no_results")
                    ? "No regions matched the prompt"
                    : "Qwen returned no usable boxes";
                setSamStatus(warning, { variant: "warn", duration: 4000 });
            }
        } catch (error) {
            const message = error?.message || error;
            setSamStatus(`Qwen error: ${message}`, { variant: "error", duration: 5000 });
            console.error("Qwen inference failed", error);
        } finally {
            qwenRequestActive = false;
            updateQwenRunButton();
        }
    }

    async function invokeQwenInfer(requestFields) {
        if (!currentImage) {
            throw new Error("No active image");
        }
        const imageNameForRequest = currentImage.name;
        const variantForRequest = samVariant;
        const preloadToken = await waitForSamPreloadIfActive(imageNameForRequest, variantForRequest);
        let payload = await buildSamImagePayload({ variantOverride: variantForRequest, preferredToken: preloadToken });
        if (imageNameForRequest && !payload.image_name) {
            payload.image_name = imageNameForRequest;
        }
        payload.sam_variant = variantForRequest;
        let resp = await fetch(`${API_ROOT}/qwen/infer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...requestFields, ...payload }),
        });
        if (resp.status === 428) {
            payload = await buildSamImagePayload({ forceBase64: true, variantOverride: variantForRequest, preferredToken: preloadToken });
            if (imageNameForRequest && !payload.image_name) {
                payload.image_name = imageNameForRequest;
            }
            payload.sam_variant = variantForRequest;
            resp = await fetch(`${API_ROOT}/qwen/infer`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ...requestFields, ...payload }),
            });
        }
        if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || resp.statusText || `HTTP ${resp.status}`);
        }
        return resp.json();
    }

    function addYoloBoxFromQwen(yoloBox, className) {
        if (!currentImage || !Array.isArray(yoloBox) || yoloBox.length < 4) {
            return null;
        }
        const [cx, cy, wNorm, hNorm] = yoloBox.map(Number);
        if ([cx, cy, wNorm, hNorm].some((val) => Number.isNaN(val))) {
            return null;
        }
        const absW = wNorm * currentImage.width;
        const absH = hNorm * currentImage.height;
        const absX = cx * currentImage.width - absW / 2;
        const absY = cy * currentImage.height - absH / 2;
        const bboxRecord = {
            x: absX,
            y: absY,
            width: absW,
            height: absH,
            marked: false,
            class: className,
            uuid: generateUUID(),
        };
        const valid = clampBbox(bboxRecord, currentImage.width, currentImage.height);
        if (!valid) {
            return null;
        }
        stampBboxCreation(bboxRecord);
        if (!bboxes[currentImage.name]) {
            bboxes[currentImage.name] = {};
        }
        if (!bboxes[currentImage.name][className]) {
            bboxes[currentImage.name][className] = [];
        }
        bboxes[currentImage.name][className].push(bboxRecord);
        return bboxRecord;
    }

    function applyQwenBoxes(boxes, className) {
        if (!currentImage || !className || !Array.isArray(boxes) || boxes.length === 0) {
            return 0;
        }
        let added = 0;
        boxes.forEach((entry) => {
            if (!entry || !entry.bbox) {
                return;
            }
            const created = addYoloBoxFromQwen(entry.bbox, className);
            if (created) {
                added += 1;
            }
        });
        if (added > 0) {
            setSamStatus(`Qwen added ${added} bbox${added === 1 ? "" : "es"} to ${className}`, { variant: "success", duration: 4500 });
        }
        return added;
    }

    function applyApiRootValue(rawValue) {
        const normalized = normalizeApiRoot(rawValue);
        if (!normalized) {
            setSettingsStatus("Enter a valid http(s) URL (e.g. http://localhost:8000)", "error");
            return;
        }
        API_ROOT = normalized;
        try {
            localStorage.setItem(API_STORAGE_KEY, normalized);
        } catch (error) {
            console.debug("Failed to persist API root", error);
        }
        if (settingsElements.apiInput) {
            settingsElements.apiInput.value = normalized;
        }
        setSettingsStatus(`Backend set to ${normalized}`, "success");
        refreshQwenStatus({ silent: true }).catch((error) => {
            console.debug("Failed to refresh Qwen status", error);
        });
        loadQwenConfig(true).catch((error) => {
            console.debug("Failed to refresh Qwen config", error);
        });
    }

    async function testApiRootCandidate(rawValue) {
        const normalized = normalizeApiRoot(rawValue);
        if (!normalized) {
            setSettingsStatus("Enter a valid http(s) URL before testing", "error");
            return;
        }
        setSettingsStatus("Testing connection…", "info");
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 5000);
            const resp = await fetch(`${normalized}/sam_slots`, { signal: controller.signal });
            clearTimeout(timeout);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            setSettingsStatus(`Connected to ${normalized}`, "success");
        } catch (error) {
            const detail = error?.name === "AbortError" ? "request timed out" : (error.message || error);
            setSettingsStatus(`Failed to reach ${normalized}: ${detail}`, "error");
        }
    }

    function handlePredictorApply(event) {
        if (event) {
            event.preventDefault();
        }
        if (!predictorElements.countInput) {
            return;
        }
        submitPredictorSettings(predictorElements.countInput.value);
    }

    function resetSamPreloadState() {
        const finishedImage = samPreloadCurrentImageName;
        const finishedVariant = samPreloadCurrentVariant;
        samPreloadCurrentImageName = null;
        samPreloadCurrentVariant = null;
        samPreloadQueuedKey = null;
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

    function isSamPreloadActiveFor(imageName, variant) {
        if (!imageName) {
            return false;
        }
        const targetKey = getSamTokenKey(imageName, variant || samVariant);
        if (!targetKey) {
            return false;
        }
        if (samPreloadQueuedKey && samPreloadQueuedKey === targetKey) {
            return true;
        }
        if (!samPreloadCurrentImageName) {
            return false;
        }
        const activeKey = getSamTokenKey(
            samPreloadCurrentImageName,
            samPreloadCurrentVariant || samVariant || variant,
        );
        return activeKey === targetKey;
    }

    function registerSamJob({ type, imageName, cleanup }) {
        const jobId = ++samJobSequence;
        const record = {
            id: jobId,
            type: type || "sam",
            imageName: imageName || null,
            version: samCancelVersion,
            cleanup: typeof cleanup === "function" ? cleanup : null,
            taskId: enqueueTask({ kind: type || "sam", imageName }),
        };
        samActiveJobs.set(jobId, record);
        return { id: jobId, version: record.version };
    }

    function completeSamJob(jobId) {
        const record = samActiveJobs.get(jobId);
        if (record && record.taskId) {
            completeTask(record.taskId);
        }
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
            if (job.taskId) {
                completeTask(job.taskId);
            }
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

    function cancelSamPreload(options = {}) {
        const preserveSet = toImageNameSet(options.preserveImages);
        samPreloadToken++;
        if (samPreloadAbortController) {
            samPreloadAbortController.abort();
            samPreloadAbortController = null;
        }
        abortSlotPreload("next", { preserveImages: preserveSet });
        abortSlotPreload("previous", { preserveImages: preserveSet });
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
        scheduleSamSlotStatusRefresh(true);
    }

    function updateSamPreloadState(checked) {
        samPreloadEnabled = !!checked;
        if (samPreloadCheckbox) {
            samPreloadCheckbox.checked = samPreloadEnabled;
        }
        if (!samPreloadEnabled) {
            samPreloadLastKey = null;
            cancelSamPreload();
            abortSlotPreload("next");
            abortSlotPreload("previous");
            return;
        }
        if (currentImage && currentImage.object) {
            ensureSamSlotsSupport().finally(() => {
                scheduleSamPreload({ force: true, immediate: true });
                if (samSlotsEnabled) {
                    scheduleSamSlotStatusRefresh(true);
                }
            });
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
        const variantSnapshot = options.variant || samVariant;
        const targetKey = getSamTokenKey(currentImage.name, variantSnapshot);
        if (!options.force && targetKey) {
            const activeKey = samPreloadCurrentImageName
                ? getSamTokenKey(samPreloadCurrentImageName, samPreloadCurrentVariant || samVariant || variantSnapshot)
                : null;
            if (targetKey === samPreloadQueuedKey || targetKey === activeKey) {
                if (options.messagePrefix) {
                    setSamStatus(`${options.messagePrefix}: continuing SAM preload`, { variant: "info", duration: 2000 });
                }
                showSamPreloadProgress();
                return;
            }
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
            variant: variantSnapshot,
            queueKey: targetKey,
            taskId: null,
        };
        if (targetImage?.meta?.name) {
            requestOptions.taskId = enqueueTask({
                kind: "sam-preload",
                imageName: targetImage.meta.name,
                detail: options.slot || "curr",
            });
        }
        if (samSlotsEnabled) {
            requestOptions.slot = options.slot || "current";
        }
        samPreloadQueuedKey = targetKey;
        samPreloadTimer = setTimeout(() => {
            samPreloadTimer = null;
            executeSamPreload(requestOptions).catch((err) => {
                console.warn("SAM preload error", err);
            });
        }, delay);
    }

    async function executeSamPreload(options) {
        if (options.queueKey && samPreloadQueuedKey === options.queueKey) {
            samPreloadQueuedKey = null;
        }
        const startTime = Date.now();
        const elapsed = startTime - (options.queuedAt || startTime);
        if (elapsed > 3000) {
            completeTask(options.taskId);
            return;
        }
        if (options.generation && options.generation < samPreloadGeneration) {
            hideSamPreloadProgress();
            resumeMultiPointQueueIfIdle();
            completeTask(options.taskId);
            return;
        }
        if (!samPreloadEnabled) {
            hideSamPreloadProgress();
            resumeMultiPointQueueIfIdle();
            completeTask(options.taskId);
            return;
        }
        const activeImage = currentImage;
        if (!activeImage || !options || activeImage !== options.imageRef || (activeImage._loadVersion || 0) !== options.version || !activeImage.object) {
            const { finishedImage, finishedVariant } = resetSamPreloadState();
            hideSamPreloadProgress();
            resolveSamPreloadWaiters(finishedImage, finishedVariant);
            completeTask(options.taskId);
            return;
        }
        const variantSnapshot = options.variant || samVariant;
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

        let releaseSlotLoading = null;
        try {
            const slotLabel = samSlotsEnabled ? (options.slot || "current") : "current";
            const requestBody = { sam_variant: variantSnapshot };
            if (samSlotsEnabled && slotLabel) {
                requestBody.slot = slotLabel;
            }
            if (samSlotsEnabled && imageSnapshot?.name) {
                requestBody.image_name = imageSnapshot.name;
            }
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
                    completeTask(options.taskId);
                    return;
                }
                requestBody.image_base64 = base64Img;
            }
            if (options.generation) {
                requestBody.preload_generation = options.generation;
            }

            if (!releaseSlotLoading && imageSnapshot?.name) {
                releaseSlotLoading = beginImageSlotLoading(imageSnapshot.name, slotLabel || "current");
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
                completeTask(options.taskId);
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
                const fallbackBody = {
                    image_base64: base64Img,
                    sam_variant: variantSnapshot,
                    preload_generation: options.generation || null,
                };
                if (samSlotsEnabled && slotLabel) {
                    fallbackBody.slot = slotLabel;
                }
                if (samSlotsEnabled && imageSnapshot?.name) {
                    fallbackBody.image_name = imageSnapshot.name;
                }
                resp = await fetch(`${API_ROOT}/sam_preload`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(fallbackBody),
                    signal: controller.signal,
                });
                if (resp.status === 409) {
                    const { finishedImage, finishedVariant } = resetSamPreloadState();
                    hideSamPreloadProgress();
                    resolveSamPreloadWaiters(finishedImage, finishedVariant);
                    completeTask(options.taskId);
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
                completeTask(options.taskId);
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
                if (slotLabel === "current" && imageSnapshot?.name) {
                    triggerNeighborSlotPreloads(imageSnapshot.name);
                    scheduleSamSlotStatusRefresh(true);
                }
                completeTask(options.taskId);
            }
        } catch (error) {
            if (error && error.name === "AbortError") {
                hideSamPreloadProgress();
                resumeMultiPointQueueIfIdle();
                completeTask(options.taskId);
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
            completeTask(options.taskId);
        } finally {
            if (typeof releaseSlotLoading === "function") {
                releaseSlotLoading();
            }
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
        bboxesFolderSelectButton = document.getElementById("bboxesSelectFolder");
        samStatusEl = document.getElementById("samStatus");
        samStatusProgressEl = document.getElementById("samStatusProgress");
        predictorElements.countInput = document.getElementById("predictorCount");
        predictorElements.applyButton = document.getElementById("predictorApply");
        predictorElements.message = document.getElementById("predictorMessage");
        predictorElements.activeCount = document.getElementById("predictorActiveCount");
        predictorElements.loadedCount = document.getElementById("predictorLoadedCount");
        predictorElements.processRam = document.getElementById("predictorProcessRam");
        predictorElements.imageRam = document.getElementById("predictorImageRam");
        predictorElements.systemFreeRam = document.getElementById("predictorSystemFreeRam");
        initQwenPanel();
        initQwenTrainingTab();
        initQwenModelTab();

        registerFileLabel(imagesSelectButton, document.getElementById("images"));
        registerFileLabel(classesSelectButton, document.getElementById("classes"));
        registerFileLabel(bboxesFolderSelectButton, document.getElementById("bboxesFolder"));
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

        if (predictorElements.applyButton) {
            predictorElements.applyButton.addEventListener("click", handlePredictorApply);
        }
        if (predictorElements.countInput) {
            predictorElements.countInput.addEventListener("keydown", (event) => {
                if (event.key === "Enter") {
                    handlePredictorApply(event);
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
        if (samSlotsEnabled && imageNameForRequest && !payload.image_name) {
            payload.image_name = imageNameForRequest;
        }
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
        const clipTaskId = enqueueTask({ kind: "clip-auto", imageName: currentImage ? currentImage.name : null, detail: bbox?.class || currentClass || null });
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
            completeTask(clipTaskId);
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
    let bboxCreationCounter = 0;

    const stampBboxCreation = (bbox) => {
        if (!bbox || typeof bbox !== "object") {
            return bbox;
        }
        if (typeof bbox.uuid === "undefined" || !bbox.uuid) {
            bbox.uuid = generateUUID();
        }
        if (typeof bbox.createdAt !== "number") {
            bbox.createdAt = ++bboxCreationCounter;
        }
        return bbox;
    };

    const findLatestCreatedBbox = (imageName) => {
        const classBuckets = bboxes[imageName];
        if (!classBuckets) {
            return null;
        }
        let latest = null;
        for (const className of Object.keys(classBuckets)) {
            const bucket = classBuckets[className];
            if (!Array.isArray(bucket) || bucket.length === 0) {
                continue;
            }
            for (let i = bucket.length - 1; i >= 0; i--) {
                const candidate = bucket[i];
                if (!candidate) {
                    continue;
                }
                const createdAt = typeof candidate.createdAt === "number" ? candidate.createdAt : -Infinity;
                if (!latest || createdAt > latest.createdAt || (createdAt === latest.createdAt && i > latest.index)) {
                    latest = {
                        className,
                        index: i,
                        bbox: candidate,
                        createdAt
                    };
                }
            }
        }
        return latest;
    };
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
            ensureBatchTweakElements();
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
                        setBboxMarkedState();
                        if (currentBbox !== null) {
                            updateBboxAfterTransform();
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
        stampBboxCreation(bbox);
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

    async function runMagicTweakForBbox(targetBbox, { updateSelection = false } = {}) {
        if (!targetBbox) {
            return false;
        }
        if (!samMode && !autoMode) {
            setSamStatus("Enable SAM or Auto Class to tweak bboxes", { variant: "warn", duration: 3000 });
            return false;
        }
        if (!currentImage || !currentImage.name) {
            return false;
        }
        if (samMode && samSlotsEnabled && samPreloadEnabled) {
            let token = getSamToken(currentImage.name, samVariant);
            if (!token) {
                const alreadyQueued = isSamPreloadActiveFor(currentImage.name, samVariant);
                setSamStatus("Waiting for SAM to load this image…", { variant: "info", duration: 0 });
                showSamPreloadProgress();
                if (!alreadyQueued) {
                    prepareSamForCurrentImage({ messagePrefix: "Preparing SAM" }).catch((err) => {
                        console.debug("prepareSamForCurrentImage (tweak) failed", err);
                    });
                }
                token = await waitForSamPreloadIfActive(currentImage.name, samVariant);
                if (!token) {
                    setSamStatus("Using fresh SAM load for this image", { variant: "info", duration: 2500 });
                } else {
                    setSamStatus(`SAM ready for ${currentImage.name}`, { variant: "success", duration: 1200 });
                }
            }
        }
        if (!targetBbox.uuid) {
            stampBboxCreation(targetBbox);
        }
        pendingApiBboxes[targetBbox.uuid] = targetBbox;
        tweakPreserveSet.add(targetBbox.uuid);
        try {
            if (samMode && autoMode) {
                await sam2BboxAutoPrompt(targetBbox);
            } else if (samMode) {
                await sam2BboxPrompt(targetBbox);
            } else if (autoMode) {
                await autoPredictNewCrop(targetBbox);
            }
            if (updateSelection) {
                setBboxMarkedState();
                if (currentBbox) {
                    updateBboxAfterTransform();
                }
            }
            return true;
        } catch (error) {
            console.warn("One-click tweak failed", error);
            setSamStatus(`Tweak failed: ${error.message || error}`, { variant: "error", duration: 4000 });
            return false;
        } finally {
            tweakPreserveSet.delete(targetBbox.uuid);
        }
    }

    async function runMagicTweakForCurrentBbox() {
        if (!currentBbox || !currentBbox.bbox) {
            setSamStatus("Select a bbox before pressing X", { variant: "warn", duration: 3000 });
            return false;
        }
        return runMagicTweakForBbox(currentBbox.bbox, { updateSelection: true });
    }

    async function runBatchTweakForCurrentCategory() {
        if (batchTweakRunning) {
            setSamStatus("Batch tweak already running", { variant: "info", duration: 2500 });
            return;
        }
        if (!currentImage || !currentImage.name) {
            setSamStatus("Load an image before batch tweaking", { variant: "warn", duration: 3000 });
            return;
        }
        if (!currentClass) {
            setSamStatus("Select a class before batch tweaking", { variant: "warn", duration: 3000 });
            return;
        }
        if (!samMode) {
            setSamStatus("Enable SAM mode to batch tweak", { variant: "warn", duration: 3000 });
            return;
        }
        const bucket = (bboxes[currentImage.name] && bboxes[currentImage.name][currentClass]) || [];
        if (!bucket.length) {
            setSamStatus("No bboxes available for this class", { variant: "warn", duration: 3000 });
            return;
        }
        batchTweakRunning = true;
        if (batchTweakElements.confirm) {
            batchTweakElements.confirm.disabled = true;
        }
        setSamStatus(`Tweaking ${bucket.length} ${currentClass} bbox${bucket.length === 1 ? "" : "es"}…`, { variant: "info", duration: 0 });
        let successCount = 0;
        try {
            for (const bbox of bucket) {
                const ok = await runMagicTweakForBbox(bbox, { updateSelection: false });
                if (ok) {
                    successCount += 1;
                }
            }
            setSamStatus(`Tweaked ${successCount}/${bucket.length} ${currentClass} bbox${bucket.length === 1 ? "" : "es"}.`, { variant: "success", duration: 3500 });
            setBboxMarkedState();
            if (currentBbox) {
                updateBboxAfterTransform();
            }
        } catch (error) {
            console.error("Batch tweak failed", error);
            setSamStatus(`Batch tweak failed: ${error.message || error}`, { variant: "error", duration: 5000 });
        } finally {
            batchTweakRunning = false;
            if (batchTweakElements.confirm) {
                batchTweakElements.confirm.disabled = false;
            }
        }
    }

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
        if (!imagesInput) {
            return;
        }
        imagesInput.addEventListener("change", async (event) => {
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

            if (imageLoadInProgress && imageLoadPromise) {
                await waitForImageLoadCompletion();
            }

            imageLoadInProgress = true;
            const loadPromise = ingestImageFiles(files, imageList, imagesInput);
            imageLoadPromise = loadPromise;
            try {
                await loadPromise;
            } finally {
                if (imageLoadPromise === loadPromise) {
                    imageLoadPromise = null;
                }
                imageLoadInProgress = false;
            }
        });
    };

    async function waitForImageLoadCompletion() {
        if (imageLoadInProgress && imageLoadPromise) {
            try {
                await imageLoadPromise;
            } catch (error) {
                console.debug("Image load completion wait failed", error);
            }
        }
    }

    async function ingestImageFiles(files, imageList, imagesInput) {
        resetImageList();
        const selectedFiles = Array.from(files || []);
        const supportedFiles = selectedFiles.filter((file) => {
            const parts = file.name.split(".");
            const ext = parts[parts.length - 1];
            return extensions.indexOf(ext) !== -1;
        });
        const total = selectedFiles.length;
        document.body.style.cursor = "wait";
        const loadingLabel = `Loading ${total} image${total === 1 ? "" : "s"}… please wait`;
        setSamStatus(loadingLabel, { variant: "info", duration: 0 });
        let fileCount = 0;
        const YIELD_EVERY = 75;

        if (supportedFiles.length > 0) {
            startIngestProgress({ phase: "images", total: supportedFiles.length });
            showBackgroundLoadModal("Images are still loading in the background. You can continue once the counter finishes.");
        }

        try {
            for (let i = 0; i < supportedFiles.length; i++) {
                const file = supportedFiles[i];
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
                option.text = file.name;
                if (fileCount === 1) {
                    option.selected = true;
                }
                imageList.appendChild(option);

                try {
                    const dim = await readImageDimensions(file);
                    images[file.name].width = dim.width;
                    images[file.name].height = dim.height;
                } catch (error) {
                    console.debug("Failed to read image dimensions", file.name, error);
                }

                incrementIngestProgress();

                if (fileCount % YIELD_EVERY === 0) {
                    await yieldToDom(0);
                }
            }
        } finally {
            document.body.style.cursor = "default";
            if (supportedFiles.length > 0) {
                stopIngestProgress();
                hideBackgroundLoadModal();
            }
        }

        if (fileCount === 0) {
            setSamStatus("No supported image files were selected.", { variant: "warn", duration: 5000 });
            imagesInput.value = "";
            return;
        }

        const firstName = getOptionImageName(imageList.options[0]);
        if (!images[firstName]) {
            setSamStatus(`Failed to stage image data for ${firstName}.`, { variant: "error", duration: 6000 });
            imagesInput.value = "";
            return;
        }
        setCurrentImage(images[firstName]);

        if (Object.keys(classes).length > 0) {
            setBboxImportEnabled(true);
        }

        setSamStatus(`Loaded ${fileCount} image${fileCount === 1 ? "" : "s"}.`, { variant: "success", duration: 3000 });
        imagesInput.value = "";
    }

    const resetImageList = () => {
        document.getElementById("imageList").innerHTML = "";
        images = {};
        bboxes = {};
        currentImage = null;
        samPreloadLastKey = null;
        cancelSamPreload();
        samTokenCache.clear();
        slotLoadingIndicators.clear();
        slotPreloadPromises.clear();
        latestSlotStatuses = [];
        applySlotStatusClasses();
        updateSlotHighlights([]);
        setBboxImportEnabled(false);
        cancelAllSamJobs({ reason: "image reset", announce: false });
        cancelPendingMultiPoint({ clearMarkers: true, removePendingBbox: true });
    };

    function setCurrentImage(image) {
        if (!image) return;
        const previousImageName = currentImage ? currentImage.name : null;
        const cancellation = cancelAllSamJobs({ reason: "image switch", imageName: previousImageName, announce: false });
        cancelPendingMultiPoint({ clearMarkers: true, removePendingBbox: true });
        const pendingImageName = image?.meta?.name || image?.name || null;
        if (previousImageName) {
            const preserveImages = pendingImageName ? [pendingImageName] : null;
            cancelSamPreload({ preserveImages });
        }
        const hasActivePreload = pendingImageName
            ? (slotPreloadPromises.has(pendingImageName) || isImageCurrentlyLoading(pendingImageName))
            : false;
        if (samPreloadEnabled && pendingImageName) {
            const statusLabel = hasActivePreload ? "Continuing SAM preload" : "Preparing SAM preload";
            setSamStatus(`${statusLabel}: ${pendingImageName}`, { variant: "info", duration: 0 });
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
                    prepareSamForCurrentImage({ messagePrefix }).catch((err) => {
                        console.debug("prepareSamForCurrentImage failed", err);
                    });
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
            prepareSamForCurrentImage({ messagePrefix, immediate: true }).catch((err) => {
                console.debug("prepareSamForCurrentImage failed", err);
            });
        }
        if (currentBbox !== null) {
            currentBbox.bbox.marked = false;
            currentBbox = null;
        }
        if (currentImage?.name) {
            syncImageSelectionToName(currentImage.name, { ensureVisible: true });
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

    const listenImageSelect = () => {
        const imageList = document.getElementById("imageList");
        imageList.addEventListener("change", () => {
            imageListIndex = imageList.selectedIndex;
            const name = getOptionImageName(imageList.options[imageListIndex]);
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
                            updateQwenClassOptions({ resetOverride: true });
                            if (Object.keys(images).length > 0) {
                                setBboxImportEnabled(true);
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
        updateQwenClassOptions({ resetOverride: true });
    };

    const setCurrentClass = () => {
        const classList = document.getElementById("classList");
        currentClass = classList.options[classList.selectedIndex].text;
        if (currentBbox !== null) {
            currentBbox.bbox.marked = false;
            currentBbox = null;
        }
        clearMultiPointAnnotations();
        syncQwenClassToCurrent();
    };

    const listenClassSelect = () => {
        const classList = document.getElementById("classList");
        classList.addEventListener("change", () => {
            classListIndex = classList.selectedIndex;
            setCurrentClass();
        });
    };

    function readFileAsTextPromise(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error || new Error("Failed to read file"));
            reader.readAsText(file);
        });
    }

    function readFileAsArrayBufferPromise(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error || new Error("Failed to read file"));
            reader.readAsArrayBuffer(file);
        });
    }

    async function processBboxFile(file) {
        const rawExtension = file.name.split(".").pop() || "";
        const extension = rawExtension.toLowerCase();
        if (extension === "txt" || extension === "xml" || extension === "json") {
            const text = await readFileAsTextPromise(file);
            storeBbox(file.name, text);
            incrementIngestProgress();
            return;
        }
        const buffer = await readFileAsArrayBufferPromise(file);
        const zip = await (typeof JSZip.loadAsync === "function"
            ? JSZip.loadAsync(buffer)
            : new JSZip().loadAsync(buffer));
        const entries = Object.values(zip.files || {}).filter((entry) => entry && !entry.dir);
        if (entries.length === 0) {
            incrementIngestProgress();
            return;
        }
        if (entries.length > 1) {
            adjustIngestTotal(entries.length - 1);
        }
        for (const entry of entries) {
            const text = await entry.async("string");
            storeBbox(entry.name, text);
            incrementIngestProgress();
        }
    }

    const listenBboxLoad = () => {
        const fileInput = document.getElementById("bboxes");
        const folderInput = document.getElementById("bboxesFolder");
        const fileButton = document.getElementById("bboxesSelect");
        const folderButton = document.getElementById("bboxesSelectFolder");
        registerFileLabel(fileButton, fileInput);
        registerFileLabel(folderButton, folderInput);
        setupBboxInputListeners(fileInput);
        setupBboxInputListeners(folderInput);
    };

    function setupBboxInputListeners(input) {
        if (!input) {
            return;
        }
        input.addEventListener("click", () => {
            input.value = "";
        });
        input.addEventListener("change", async (event) => {
            if (imageLoadInProgress) {
                setSamStatus("Still loading images — please wait before importing bboxes.", { variant: "info", duration: 3000 });
                await waitForImageLoadCompletion();
            }
            const files = event.target.files;
            if (files && files.length > 0) {
                resetBboxes();
                bboxImportCounterActive = true;
                startIngestProgress({ phase: "bboxes", total: files.length, extraLabel: "UUIDs" });
                try {
                    const tasks = [];
                    for (let i = 0; i < files.length; i++) {
                        tasks.push(
                            processBboxFile(files[i]).catch((err) => {
                                console.error("Failed to import bbox file", files[i]?.name, err);
                                setSamStatus(`Failed to import ${files[i]?.name}: ${err.message || err}`, { variant: "error", duration: 5000 });
                            })
                        );
                    }
                    await Promise.all(tasks);
                } finally {
                    bboxImportCounterActive = false;
                    stopIngestProgress();
                }
            }
            input.value = "";
        });
    }

    const resetBboxes = () => {
        bboxes = {};
    };

    const storeBbox = (filename, text) => {
        // same storeBbox logic you had before
        let image = null;
        let bbox = null;
        const rawExtension = filename.split(".").pop() || "";
        const extension = rawExtension.toLowerCase();
        const baseName = rawExtension ? filename.slice(0, -(rawExtension.length + 1)) : filename;
        if (extension === "txt" || extension === "xml") {
            for (let i = 0; i < extensions.length; i++) {
                const imageName = `${baseName}.${extensions[i]}`;
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
                                    const bboxRecord = {
                                        x: Math.floor(x),
                                        y: Math.floor(y),
                                        width: Math.floor(width),
                                        height: Math.floor(height),
                                        marked: false,
                                        class: className
                                    };
                                    stampBboxCreation(bboxRecord);
                                    bbox[className].push(bboxRecord);
                                    noteImportedBbox();
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
                                    const bboxRecord = {
                                        x: parseInt(bndBoxX),
                                        y: parseInt(bndBoxY),
                                        width: parseInt(bndBoxMaxX) - parseInt(bndBoxX),
                                        height: parseInt(bndBoxMaxY) - parseInt(bndBoxY),
                                        marked: false,
                                        class: className
                                    };
                                    stampBboxCreation(bboxRecord);
                                    bbox[className].push(bboxRecord);
                                    noteImportedBbox();
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
                        const bboxRecord = {
                            x: bboxX,
                            y: bboxY,
                            width: bboxWidth,
                            height: bboxHeight,
                            marked: false,
                            class: className
                        };
                        stampBboxCreation(bboxRecord);
                        bbox[className].push(bboxRecord);
                        noteImportedBbox();
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
            const targetTag = (event.target?.tagName || "").toLowerCase();
            const inputType = (event.target?.getAttribute && event.target.getAttribute("type")) ? event.target.getAttribute("type").toLowerCase() : "";
            const isTextualInput = targetTag === "textarea"
                || (targetTag === "input" && !["checkbox", "radio", "button", "range", "color"].includes(inputType));
            if (isTextualInput || event.target?.isContentEditable) {
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

            if (!event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey && (key === 88 || event.key === "x" || event.key === "X")) {
                event.preventDefault();
                handleXHotkeyPress();
                return;
            }

            const plainDeleteHotkey = !event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey
                && (key === 87 || event.key === "w" || event.key === "W");
            if (key === 8 || (key === 46 && event.metaKey === true) || plainDeleteHotkey) {
                if (currentBbox !== null) {
                    bboxes[currentImage.name][currentBbox.bbox.class].splice(currentBbox.index, 1);
                    currentBbox = null;
                    document.body.style.cursor = "default";
                }
                event.preventDefault();
            }
            if (!event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey && (key === 81 || event.key === "q" || event.key === "Q")) {
                let removed = false;
                if (currentImage && bboxes[currentImage.name]) {
                    const latest = findLatestCreatedBbox(currentImage.name);
                    if (latest) {
                        const bucket = bboxes[currentImage.name][latest.className];
                        if (bucket) {
                            const spliceResult = bucket.splice(latest.index, 1);
                            if (bucket.length === 0) {
                                delete bboxes[currentImage.name][latest.className];
                            }
                            if (spliceResult.length > 0) {
                                removed = true;
                                if (currentBbox && currentBbox.bbox === spliceResult[0]) {
                                    currentBbox = null;
                                    document.body.style.cursor = "default";
                                }
                            }
                        }
                    }
                }
                if (removed) {
                    event.preventDefault();
                }
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
                    const imageName = getOptionImageName(imageList.options[imageListIndex]);
                    if (imageName && images[imageName]) {
                        setCurrentImage(images[imageName]);
                    }
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
                    const imageName = getOptionImageName(imageList.options[imageListIndex]);
                    if (imageName && images[imageName]) {
                        setCurrentImage(images[imageName]);
                    }
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

    async function handleMagicTweakHotkey() {
        if (magicTweakRunning) {
            return;
        }
        magicTweakRunning = true;
        try {
            await runMagicTweakForCurrentBbox();
        } finally {
            magicTweakRunning = false;
        }
    }

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
