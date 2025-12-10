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

    function setAgentStatus(text, tone = "info") {
        if (!agentElements.status) return;
        agentElements.status.textContent = text || "";
        agentElements.status.className = `training-message ${tone}`;
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
    const TAB_SAM3_TRAIN = "sam3-train";
    const TAB_AGENT_MINING = "agent-mining";
    const TAB_PROMPT_HELPER = "prompt-helper";
    const TAB_DATASETS = "datasets";
    const TAB_SAM3_PROMPT_MODELS = "sam3-prompt-models";
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
    let sam3TrainUiInitialized = false;
    let segBuilderUiInitialized = false;
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
        sam3TrainButton: null,
        agentMiningButton: null,
        promptHelperButton: null,
        sam3PromptModelsButton: null,
        datasetsButton: null,
        activeButton: null,
        qwenButton: null,
        predictorsButton: null,
        settingsButton: null,
        labelingPanel: null,
        trainingPanel: null,
        qwenTrainPanel: null,
        sam3TrainPanel: null,
        agentMiningPanel: null,
        promptHelperPanel: null,
        sam3PromptModelsPanel: null,
        datasetsPanel: null,
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
    const sam3TextElements = {
        panel: null,
        promptInput: null,
        thresholdInput: null,
        maskThresholdInput: null,
        maxResultsInput: null,
        runButton: null,
        autoButton: null,
        similarityButton: null,
        similarityRow: null,
        similarityThresholdInput: null,
        status: null,
        classSelect: null,
        minSizeInput: null,
        maxPointsInput: null,
        epsilonInput: null,
    };
    const sam3RecipeElements = {
        fileInput: null,
        applyButton: null,
        status: null,
        presetSelect: null,
        presetNameInput: null,
        presetSaveButton: null,
        presetLoadButton: null,
        presetRefreshButton: null,
    };
    const sam3RecipeState = {
        recipe: null,
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
        refreshButton: null,
    };
    let qwenAvailable = false;
    let qwenRequestActive = false;
    let sam3TextRequestActive = false;
    let sam3SimilarityRequestActive = false;
    let qwenClassOverride = false;
    let qwenAdvancedVisible = false;
    const qwenModelState = {
        models: [],
        activeId: "default",
        activeMetadata: DEFAULT_QWEN_METADATA,
    };
    let sam3TextUiInitialized = false;

    let settingsUiInitialized = false;
    const backendFuzzerElements = {
        runButton: null,
        status: null,
        log: null,
        includeQwen: null,
        includeSam3: null,
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
        devicesInput: null,
        loraRankInput: null,
        loraAlphaInput: null,
        loraDropoutInput: null,
        patienceInput: null,
        maxImageDimInput: null,
        maxDetectionsInput: null,
        datasetModeUpload: null,
        datasetModeCached: null,
        datasetSelect: null,
        datasetRefresh: null,
        datasetDelete: null,
        datasetSummary: null,
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
        epochDetail: null,
        message: null,
        summary: null,
        log: null,
        historyContainer: null,
        lossCanvas: null,
        chartStatus: null,
        chartSmoothing: null,
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
    chartSmoothing: 15,
    lastJobSnapshot: null,
};

    const qwenDatasetState = {
        items: [],
        selectedId: null,
    };

    const sam3TrainElements = {
        datasetSelect: null,
        datasetSummary: null,
        datasetRefresh: null,
        datasetConvert: null,
        runName: null,
        trainBatch: null,
        valBatch: null,
        trainWorkers: null,
        valWorkers: null,
        epochs: null,
        resolution: null,
        lrScale: null,
        gradAccum: null,
        valFreq: null,
        targetEpochSize: null,
        warmupSteps: null,
        schedulerTimescale: null,
        balanceStrategy: null,
        balancePower: null,
        balanceClip: null,
        balanceBeta: null,
        balanceGamma: null,
        freezeLanguage: null,
        languageLr: null,
        promptVariants: null,
        promptRandomize: null,
        logAll: null,
        valScoreThresh: null,
        valMaxDets: null,
        segHead: null,
        segTrain: null,
        startButton: null,
        cancelButton: null,
        statusText: null,
        progressFill: null,
        message: null,
        summary: null,
        balanceDescription: null,
        log: null,
        history: null,
        activateButton: null,
    };

const sam3TrainState = {
    datasets: [],
    selectedId: null,
    activeJobId: null,
    pollHandle: null,
    lastJobSnapshot: null,
    latestCheckpoint: null,
    trendAlpha: 0.05,
    valMetrics: [],
};

    const sam3StorageElements = {
        list: null,
        refresh: null,
    };

    const sam3StorageState = {
        items: [],
    };

    const promptHelperElements = {
        datasetSelect: null,
        datasetRefresh: null,
        datasetSummary: null,
        sampleSize: null,
        maxSynonyms: null,
        scoreThresh: null,
        maxDets: null,
        iouThresh: null,
        seed: null,
        useQwen: null,
        generateButton: null,
        evaluateButton: null,
        presetName: null,
        presetSaveBtn: null,
        presetSelect: null,
        presetLoadBtn: null,
        status: null,
        summary: null,
        prompts: null,
        results: null,
        logs: null,
        message: null,
        applyButton: null,
    };

    const promptHelperState = {
        datasets: [],
        selectedId: null,
        activeJobId: null,
        pollHandle: null,
        lastJob: null,
        suggestions: [],
        promptsByClass: {},
        presets: [],
    };

    const promptSearchElements = {
        sampleSize: null,
        negatives: null,
        precisionFloor: null,
        scoreThresh: null,
        maxDets: null,
        iouThresh: null,
        seed: null,
        runButton: null,
        status: null,
        logs: null,
        results: null,
        message: null,
        classSelect: null,
    };

    const promptSearchState = {
        activeJobId: null,
        pollHandle: null,
        lastJob: null,
    };

    const promptRecipeElements = {
        classSelect: null,
        sampleSize: null,
        negatives: null,
        thresholds: null,
        maxDets: null,
        iouThresh: null,
        seed: null,
        expandCount: null,
        expandButton: null,
        runButton: null,
        status: null,
        logs: null,
        results: null,
        message: null,
    };
    const agentElements = {
        datasetSelect: null,
        datasetRefresh: null,
        datasetSummary: null,
        valPercent: null,
        thresholds: null,
        maskThreshold: null,
        maxResults: null,
        minSize: null,
        simplifyEps: null,
        maxWorkers: null,
        workersPerGpu: null,
        exemplars: null,
        clusterExemplars: null,
        clipGuard: null,
        similarityScore: null,
        classesInput: null,
        classHints: null,
        testMode: null,
        trainLimit: null,
        valLimit: null,
        runButton: null,
        refreshButton: null,
        cancelButton: null,
        status: null,
        results: null,
        recipeSelect: null,
        recipeRefresh: null,
        recipeLoad: null,
        recipeDownload: null,
        recipeApply: null,
        recipeImageId: null,
        recipeImport: null,
        recipeFile: null,
        recipeDetails: null,
        logs: null,
        progressFill: null,
    };

    const promptRecipeState = {
        activeJobId: null,
        pollHandle: null,
        lastJob: null,
    };
    const agentState = {
        lastJob: null,
        datasetsById: {},
        loadedRecipe: null,
        recipeClassOverride: null,
        pollTimer: null,
        pollInFlight: false,
    };

    let promptHelperInitialized = false;

    const segBuilderElements = {
        datasetSelect: null,
        datasetSummary: null,
        outputName: null,
        samVariant: null,
        startButton: null,
        refreshButton: null,
        jobsRefresh: null,
        jobsContainer: null,
        message: null,
    };

    const segBuilderState = {
        datasets: [],
        selectedId: null,
        jobs: [],
        lastSeenJob: {},
    };

    const datasetManagerElements = {
        uploadFile: null,
        uploadName: null,
        uploadType: null,
        uploadBtn: null,
        uploadMessage: null,
        uploadCurrentBtn: null,
        uploadCurrentSummary: null,
        refreshBtn: null,
        list: null,
    };

    const datasetManagerState = {
        datasets: [],
        uploading: false,
    };

    function setDatasetUploadMessage(text, tone) {
        if (!datasetManagerElements.uploadMessage) return;
        datasetManagerElements.uploadMessage.textContent = text || "";
        datasetManagerElements.uploadMessage.className = `training-message ${tone || ""}`;
    }

    function renderDatasetList(list) {
        datasetManagerState.datasets = Array.isArray(list) ? list : [];
        const container = datasetManagerElements.list;
        if (container) {
            container.innerHTML = "";
            if (!datasetManagerState.datasets.length) {
                const empty = document.createElement("div");
                empty.className = "training-history-item";
                empty.textContent = "No datasets yet. Upload one to get started.";
                container.appendChild(empty);
            } else {
                datasetManagerState.datasets.forEach((entry) => {
                    const item = document.createElement("div");
                    item.className = "training-history-item";
                    const header = document.createElement("div");
                    header.className = "training-history-row";
                    const title = document.createElement("div");
                    title.className = "training-history-title";
                    title.textContent = entry.label || entry.id;
                    const badge = document.createElement("span");
                    badge.className = "badge";
                    badge.textContent = (entry.type || "bbox").toUpperCase();
                    header.appendChild(title);
                    header.appendChild(badge);
                    const meta = document.createElement("div");
                    meta.className = "training-help";
                    const parts = [];
                    if (entry.source) parts.push(entry.source);
                    const counts = [];
                    if (entry.image_count) counts.push(`${entry.image_count} img`);
                    if (entry.train_count) counts.push(`train ${entry.train_count}`);
                    if (entry.val_count) counts.push(`val ${entry.val_count}`);
                    if (counts.length) parts.push(counts.join(" / "));
                    meta.textContent = parts.join(" • ") || "Dataset ready";
                    item.appendChild(header);
                    item.appendChild(meta);
                    container.appendChild(item);
                });
            }
        }
        renderSegBuilderDatasets(datasetManagerState.datasets);
    }

    async function refreshDatasetList() {
        try {
            const resp = await fetch(`${API_ROOT}/datasets`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            renderDatasetList(data);
        } catch (err) {
            console.error("Failed to refresh datasets", err);
            setDatasetUploadMessage(`Failed to load datasets: ${err.message || err}`, "error");
        }
    }

    async function uploadDatasetZip() {
        if (!datasetManagerElements.uploadFile || !datasetManagerElements.uploadFile.files.length) {
            setDatasetUploadMessage("Choose a zip file first.", "warn");
            return;
        }
        const formData = new FormData();
        formData.append("file", datasetManagerElements.uploadFile.files[0]);
        if (datasetManagerElements.uploadName && datasetManagerElements.uploadName.value.trim()) {
            formData.append("dataset_id", datasetManagerElements.uploadName.value.trim());
        }
        if (datasetManagerElements.uploadType && datasetManagerElements.uploadType.value) {
            formData.append("dataset_type", datasetManagerElements.uploadType.value);
        }
        datasetManagerState.uploading = true;
        setDatasetUploadMessage("Uploading dataset…", "info");
        try {
            const resp = await fetch(`${API_ROOT}/datasets/upload`, { method: "POST", body: formData });
            const detail = await resp.text();
            if (!resp.ok) {
                throw new Error(detail || `HTTP ${resp.status}`);
            }
            const data = detail ? JSON.parse(detail) : {};
            setDatasetUploadMessage(`Uploaded ${data.label || data.id || "dataset"}.`, "success");
            if (datasetManagerElements.uploadFile) {
                datasetManagerElements.uploadFile.value = "";
            }
            await refreshDatasetList();
        } catch (err) {
            console.error("Dataset upload failed", err);
            setDatasetUploadMessage(err.message || "Dataset upload failed", "error");
        } finally {
            datasetManagerState.uploading = false;
        }
    }

    async function uploadCurrentDatasetToCache() {
        try {
            if (datasetType === "seg") {
                setDatasetUploadMessage("Uploading current dataset is only supported for bbox mode right now.", "warn");
                return;
            }
            setDatasetUploadMessage("Packaging current dataset…", "info");
            const result = await uploadQwenDatasetStream();
            const label = result?.run_name || "dataset";
            setDatasetUploadMessage(`Uploaded ${label} from the labeling tab.`, "success");
            if (datasetManagerElements.uploadCurrentSummary) {
                datasetManagerElements.uploadCurrentSummary.textContent = `Cached as ${label}`;
            }
            await refreshDatasetList();
        } catch (err) {
            console.error("Upload current dataset failed", err);
            setDatasetUploadMessage(err.message || "Failed to upload current dataset", "error");
        }
    }

    async function initDatasetManagerTab() {
        if (datasetManagerElements.uploadBtn) {
            return;
        }
        datasetManagerElements.uploadFile = document.getElementById("datasetUploadFile");
        datasetManagerElements.uploadName = document.getElementById("datasetUploadName");
        datasetManagerElements.uploadType = document.getElementById("datasetUploadType");
        datasetManagerElements.uploadBtn = document.getElementById("datasetUploadBtn");
        datasetManagerElements.uploadMessage = document.getElementById("datasetUploadMessage");
        datasetManagerElements.uploadCurrentBtn = document.getElementById("datasetUploadCurrentBtn");
        datasetManagerElements.uploadCurrentSummary = document.getElementById("datasetUploadCurrentSummary");
        datasetManagerElements.refreshBtn = document.getElementById("datasetListRefresh");
        datasetManagerElements.list = document.getElementById("datasetList");
        if (datasetManagerElements.uploadBtn) {
            datasetManagerElements.uploadBtn.addEventListener("click", () => uploadDatasetZip());
        }
        if (datasetManagerElements.uploadCurrentBtn) {
            datasetManagerElements.uploadCurrentBtn.addEventListener("click", () => uploadCurrentDatasetToCache());
        }
        if (datasetManagerElements.refreshBtn) {
            datasetManagerElements.refreshBtn.addEventListener("click", () => refreshDatasetList());
        }
        initSegBuilderUi();
        await refreshDatasetList();
        await refreshSegBuilderJobs();
    }

    const sam3PromptElements = {
        select: null,
        refresh: null,
        summary: null,
        message: null,
        activate: null,
    };

    const sam3PromptState = {
        models: [],
        selected: null,
    };

    function initSam3PromptModelsUi() {
        sam3PromptElements.select = document.getElementById("sam3PromptModelSelect");
        sam3PromptElements.refresh = document.getElementById("sam3PromptRefresh");
        sam3PromptElements.summary = document.getElementById("sam3PromptModelSummary");
        sam3PromptElements.message = document.getElementById("sam3PromptMessage");
        sam3PromptElements.activate = document.getElementById("sam3PromptActivate");
        if (sam3PromptElements.select) {
            sam3PromptElements.select.addEventListener("change", () => updateSam3PromptSummary());
        }
        if (sam3PromptElements.refresh) {
            sam3PromptElements.refresh.addEventListener("click", () => refreshSam3PromptModels());
        }
        if (sam3PromptElements.activate) {
            sam3PromptElements.activate.addEventListener("click", () => activateSam3PromptModel());
        }
        refreshSam3PromptModels();
    }

    async function refreshSam3PromptModels() {
        if (!sam3PromptElements.select) return;
        try {
            const resp = await fetch(`${API_ROOT}/sam3/models/available?variant=all&promoted_only=true`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            sam3PromptState.models = Array.isArray(data) ? data : [];
            sam3PromptState.models.sort((a, b) => {
                const pa = a.promoted ? 0 : 1;
                const pb = b.promoted ? 0 : 1;
                if (pa !== pb) return pa - pb;
                const va = a.variant || "";
                const vb = b.variant || "";
                if (va !== vb) return va.localeCompare(vb);
                return (a.id || "").localeCompare(b.id || "");
            });
            sam3PromptElements.select.innerHTML = "";
            sam3PromptState.models.forEach((m, idx) => {
                const opt = document.createElement("option");
                opt.value = m.key || m.path || m.id || `model-${idx}`;
                const sizeText = Number.isFinite(m.size_bytes) ? ` – ${formatBytes(m.size_bytes)}` : "";
                opt.textContent = `${m.id || `run ${idx + 1}`} [${m.variant || "sam3"}]${m.promoted ? " (promoted)" : ""}${sizeText}`;
                sam3PromptElements.select.appendChild(opt);
            });
            if (sam3PromptState.models.length) {
                const first = sam3PromptState.models[0];
                sam3PromptElements.select.value = first.key || first.path || first.id || sam3PromptElements.select.options[0].value;
                updateSam3PromptSummary();
            } else {
                updateSam3PromptSummary();
            }
        } catch (err) {
            console.error("Failed to load SAM3 prompt models", err);
            setSam3PromptMessage(`Load failed: ${err.message || err}`, "error");
        }
    }

    function updateSam3PromptSummary() {
        if (!sam3PromptElements.summary) return;
        const path = sam3PromptElements.select ? sam3PromptElements.select.value : null;
        const entry = sam3PromptState.models.find((m) => (m.key || m.path || m.id) === path);
        if (!entry) {
            sam3PromptElements.summary.textContent = "No model selected.";
            return;
        }
        const parts = [];
        if (entry.promoted) parts.push("promoted");
        if (Number.isFinite(entry.size_bytes)) parts.push(formatBytes(entry.size_bytes));
        if (entry.run_path) parts.push(`run: ${entry.run_path}`);
        sam3PromptElements.summary.textContent = parts.length ? parts.join(" • ") : "";
    }

    function setSam3PromptMessage(text, tone = "info") {
        if (!sam3PromptElements.message) return;
        sam3PromptElements.message.textContent = text || "";
        sam3PromptElements.message.className = `training-message ${tone}`;
    }

    async function activateSam3PromptModel() {
        if (!sam3PromptElements.select) return;
        const path = sam3PromptElements.select.value;
        const entry = sam3PromptState.models.find((m) => (m.key || m.path || m.id) === path);
        if (!entry) {
            setSam3PromptMessage("Select a model first.", "warn");
            return;
        }
        const payload = {
            checkpoint_path: entry.path || null,
            enable_segmentation: false,
            label: `prompt:${(entry.id || entry.path || "sam3").toString().split("/").pop()}`,
        };
        try {
            const resp = await fetch(`${API_ROOT}/sam3/models/activate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            setSam3PromptMessage("SAM3 prompt model activated.", "success");
        } catch (err) {
            console.error("Activate SAM3 prompt model failed", err);
            setSam3PromptMessage(`Activate failed: ${err.message || err}`, "error");
        }
    }

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

    function updateBalanceParamVisibility(strategy) {
        const chosen = strategy || (sam3TrainElements.balanceStrategy && sam3TrainElements.balanceStrategy.value) || "none";
        const rows = document.querySelectorAll(".sam3-balance-param");
        rows.forEach((row) => {
            const param = row.dataset ? row.dataset.param : null;
            let show = false;
            if (param === "power") {
                show = ["inv_sqrt", "clipped_inv"].includes(chosen);
            } else if (param === "clip") {
                show = chosen === "clipped_inv";
            } else if (param === "beta") {
                show = chosen === "effective_num";
            } else if (param === "gamma") {
                show = chosen === "focal";
            }
            row.style.display = show ? "" : "none";
        });
        if (sam3TrainElements.balanceDescription) {
            let desc = "";
            if (chosen === "inv_sqrt") {
                desc = "Weights = sum(1 / freq^power). Power < 1 gives mild up-weighting of rare classes (default power 0.5).";
            } else if (chosen === "clipped_inv") {
                desc = "Inverse-frequency with a cap: weight ∝ 1/freq^power, then clipped so max/min ≤ clip ratio.";
            } else if (chosen === "effective_num") {
                desc = "Effective number of samples: weight ∝ (1-β)/(1-β^n). Higher β (e.g., 0.99–0.999) boosts rare classes smoothly.";
            } else if (chosen === "focal") {
                desc = "Focal-style sampling: weight ∝ (freq / max_freq)^(-γ). Higher γ boosts rare/low-freq classes.";
            } else {
                desc = "Uniform sampling (no class rebalance).";
            }
            sam3TrainElements.balanceDescription.textContent = desc;
        }
    }

    function useCachedQwenDataset() {
        return Boolean(qwenTrainElements.datasetModeCached?.checked);
    }

    function getSelectedQwenDataset() {
        const id = qwenDatasetState.selectedId;
        if (!id) {
            return null;
        }
        return qwenDatasetState.items.find((entry) => entry.id === id) || null;
    }

    function selectQwenDatasetById(datasetId) {
        if (!datasetId) {
            return;
        }
        qwenDatasetState.selectedId = datasetId;
        if (qwenTrainElements.datasetSelect) {
            qwenTrainElements.datasetSelect.value = datasetId;
        }
        updateQwenDatasetSummary();
    }

    function updateQwenDatasetSummary() {
        const summaryEl = qwenTrainElements.datasetSummary;
        if (!summaryEl) {
            return;
        }
        if (useCachedQwenDataset()) {
            const entry = getSelectedQwenDataset();
            if (entry) {
                const context = entry.context ? ` Context: ${entry.context}` : "";
                summaryEl.textContent = `Using cached dataset "${entry.label}" (${entry.image_count || 0} images, train ${entry.train_count || 0} / val ${entry.val_count || 0}).${context}`;
            } else {
                summaryEl.textContent = "Select a cached dataset or switch back to uploading the current dataset.";
            }
        } else {
            summaryEl.textContent = "We'll upload the dataset from the labeling tab and cache it automatically.";
        }
    }

    function setQwenDatasetModeState() {
        const useCached = useCachedQwenDataset();
        if (qwenTrainElements.datasetSelect) {
            qwenTrainElements.datasetSelect.disabled = !useCached || !qwenDatasetState.items.length;
        }
        if (qwenTrainElements.datasetRefresh) {
            qwenTrainElements.datasetRefresh.disabled = !useCached;
        }
        if (qwenTrainElements.datasetDelete) {
            qwenTrainElements.datasetDelete.disabled = !useCached || !qwenDatasetState.items.length;
        }
        if (useCached && !qwenDatasetState.items.length) {
            loadQwenDatasetList(true).catch((error) => console.error("Failed to load cached datasets", error));
        }
        updateQwenDatasetSummary();
    }

    function populateQwenDatasetSelect() {
        const select = qwenTrainElements.datasetSelect;
        if (!select) {
            return;
        }
        select.innerHTML = "";
        if (!qwenDatasetState.items.length) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "No cached datasets";
            select.appendChild(option);
            select.disabled = true;
        } else {
            qwenDatasetState.items.forEach((entry) => {
                const option = document.createElement("option");
                option.value = entry.id;
                option.textContent = `${entry.label || entry.id} (${entry.image_count || 0} images)`;
                select.appendChild(option);
            });
            if (!qwenDatasetState.selectedId || !qwenDatasetState.items.some((entry) => entry.id === qwenDatasetState.selectedId)) {
                qwenDatasetState.selectedId = qwenDatasetState.items[0].id;
            }
            select.disabled = false;
            select.value = qwenDatasetState.selectedId;
        }
        if (qwenTrainElements.datasetDelete) {
            qwenTrainElements.datasetDelete.disabled = !useCachedQwenDataset() || !qwenDatasetState.items.length;
        }
        if (qwenTrainElements.datasetRefresh) {
            qwenTrainElements.datasetRefresh.disabled = !useCachedQwenDataset();
        }
        updateQwenDatasetSummary();
    }

    async function loadQwenDatasetList(force = false) {
        if (!force && qwenDatasetState.items.length) {
            populateQwenDatasetSelect();
            return qwenDatasetState.items;
        }
        try {
            const resp = await fetch(`${API_ROOT}/qwen/datasets`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const data = await resp.json();
            qwenDatasetState.items = Array.isArray(data) ? data : [];
            populateQwenDatasetSelect();
            return qwenDatasetState.items;
        } catch (error) {
            console.error("Failed to load cached Qwen datasets", error);
            if (qwenTrainElements.datasetSummary) {
                qwenTrainElements.datasetSummary.textContent = `Unable to load cached datasets: ${error.message || error}`;
            }
            return [];
        }
    }

    async function handleQwenDatasetDelete() {
        const entry = getSelectedQwenDataset();
        if (!entry) {
            setQwenTrainMessage("Select a cached dataset to delete.", "warn");
            return;
        }
        if (!window.confirm(`Delete cached dataset "${entry.label}"? This cannot be undone.`)) {
            return;
        }
        try {
            const resp = await fetch(`${API_ROOT}/qwen/datasets/${encodeURIComponent(entry.id)}`, {
                method: "DELETE",
            });
            if (!resp.ok) {
                const detail = await resp.text();
                throw new Error(detail || `HTTP ${resp.status}`);
            }
            setQwenTrainMessage(`Deleted cached dataset "${entry.label}".`, "success");
            qwenDatasetState.selectedId = null;
            await loadQwenDatasetList(true);
        } catch (error) {
            console.error("Failed to delete Qwen dataset", error);
            setQwenTrainMessage(error.message || "Failed to delete dataset", "error");
        }
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
            if (data.clip_ready) {
                setActiveMessage("CLIP is ready for auto-labeling.", "success");
            } else if (data.clip_error && String(data.clip_error).includes("numpy._core")) {
                setActiveMessage("CLIP classifier failed to load: numpy version mismatch. Please retrain CLIP in this environment or re-export with the current NumPy.", "error");
            } else if (data.clip_error) {
                setActiveMessage(`CLIP classifier not ready: ${data.clip_error}`, "error");
            } else {
                setActiveMessage("CLIP classifier is not ready. Load a model to enable auto-labeling.", "error");
            }
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

const sam3LossState = {
    jobId: null,
    avgPoints: [],
    instPoints: [],
    trendPoints: [],
    rawPoints: [],
    lastMetricCount: 0,
    chart: null, // { ctx }
};

function renderSam3ValMetrics() {
    const container = sam3TrainElements.valMetrics;
    if (!container) return;
    const vals = sam3TrainState.valMetrics || [];
    if (!vals.length) {
        container.textContent = "No validation metrics yet.";
        return;
    }
    const valsSorted = [...vals].sort((a, b) => {
        const ea = Number.isFinite(a.epoch) ? a.epoch : 0;
        const eb = Number.isFinite(b.epoch) ? b.epoch : 0;
        return eb - ea; // latest first
    });
    const latest = valsSorted[0];
    const best = valsSorted.reduce((bestSoFar, entry) => {
        if (Number.isFinite(entry.ap) && (!bestSoFar || entry.ap > bestSoFar.ap)) {
            return entry;
        }
        return bestSoFar;
    }, null);
    const rows = valsSorted
        .map((entry, idx) => {
            const ep = Number.isFinite(entry.epoch) ? entry.epoch : `#${idx + 1}`;
            const ap = Number.isFinite(entry.ap) ? entry.ap.toFixed(3) : "–";
            const ap50 = Number.isFinite(entry.ap50) ? entry.ap50.toFixed(3) : "–";
            const ap75 = Number.isFinite(entry.ap75) ? entry.ap75.toFixed(3) : "–";
            const ar10 = Number.isFinite(entry.ar10) ? entry.ar10.toFixed(3) : "–";
            const ar100 = Number.isFinite(entry.ar100) ? entry.ar100.toFixed(3) : "–";
            const cls = idx === 0 ? "highlight" : "";
            return `<tr class="${cls}"><td>${ep}</td><td>${ap}</td><td>${ap50}</td><td>${ap75}</td><td>${ar10}</td><td>${ar100}</td></tr>`;
        })
        .join("");
    const latestLabel = Number.isFinite(latest.epoch) ? `epoch ${latest.epoch}` : `validation #${vals.length}`;
    const bestLabel =
        best && Number.isFinite(best.epoch)
            ? `epoch ${best.epoch} (best AP)`
            : best
              ? "best AP"
              : "n/a";
    const bestAp = best && Number.isFinite(best.ap) ? best.ap.toFixed(3) : "–";
    const bestAp50 = best && Number.isFinite(best.ap50) ? best.ap50.toFixed(3) : "–";
    const bestAp75 = best && Number.isFinite(best.ap75) ? best.ap75.toFixed(3) : "–";
    container.innerHTML = `
        <div class="training-help">
            COCO bbox metrics on the validation set (higher is better).
            AP is averaged over IoU 0.50–0.95; AP50/AP75 are stricter IoU thresholds; AR10/AR100 are recall with 10/100 detections per image.
        </div>
        <div class="training-help">
            Latest ${latestLabel}: AP ${Number.isFinite(latest.ap) ? latest.ap.toFixed(3) : "–"} • AP50 ${
        Number.isFinite(latest.ap50) ? latest.ap50.toFixed(3) : "–"
    } • AP75 ${Number.isFinite(latest.ap75) ? latest.ap75.toFixed(3) : "–"}
            ${best ? `<br/>Best ${bestLabel}: AP ${bestAp} • AP50 ${bestAp50} • AP75 ${bestAp75}` : ""}
        </div>
        <table class="metrics-table">
            <thead><tr><th>Epoch</th><th>AP (0.50–0.95)</th><th>AP50</th><th>AP75</th><th>AR10</th><th>AR100</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

const sam3EtaState = {
    startTime: null,
};

function initSam3LossChart() {
    if (!sam3TrainElements.lossCanvas || sam3LossState.chart) return;
    const ctx = sam3TrainElements.lossCanvas.getContext("2d");
    sam3LossState.chart = { ctx };
}

function resetSam3LossChart(jobId = null) {
    sam3LossState.jobId = jobId;
    sam3LossState.avgPoints = [];
    sam3LossState.instPoints = [];
    sam3LossState.trendPoints = [];
    sam3LossState.rawPoints = [];
    sam3LossState.lastMetricCount = 0;
    const canvas = sam3TrainElements.lossCanvas;
    if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
            const width = canvas.width || canvas.clientWidth || 0;
            const height = canvas.height || canvas.clientHeight || 0;
            ctx.clearRect(0, 0, width, height);
        }
    }
}

function parseSam3LossPair(line) {
    // Expect log lines like "Losses/train_all_loss: 9.58e+01 (1.23e+02)"
    const match = line.match(/Losses\/train_all_loss:\s*([0-9.+-eE]+)(?:\s*\(\s*([0-9.+-eE]+)\s*\))?/);
    if (!match) return null;
    const inst = Number(match[1]);
    const avg = match[2] !== undefined ? Number(match[2]) : null;
    return {
        instant: Number.isFinite(inst) ? inst : null,
        average: Number.isFinite(avg) ? avg : null,
    };
}

function formatEta(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins <= 0) return `${secs}s remaining`;
    if (mins < 60) return `${mins}m ${secs.toString().padStart(2, "0")}s remaining`;
    const hours = Math.floor(mins / 60);
    const remMins = mins % 60;
    return `${hours}h ${remMins.toString().padStart(2, "0")}m remaining`;
}

function resetSam3Eta() {
    sam3EtaState.startTime = null;
}

function computeSam3Progress(job) {
    const fallback = Number.isFinite(job?.progress) ? job.progress : 0;
    const metrics = Array.isArray(job?.metrics) ? job.metrics : [];
    if (!metrics.length) return Math.max(0, Math.min(1, fallback));
    const last = metrics[metrics.length - 1] || {};
    // Validation progress (if present) overrides train-derived progress
    const lastVal = [...metrics].reverse().find(
        (m) => m && m.phase === "val" && Number.isFinite(m.val_step) && Number.isFinite(m.val_total),
    );
    if (lastVal) {
        const valStep = Number(lastVal.val_step);
        const valTotal = Math.max(1, Number(lastVal.val_total));
        const valFrac = Math.max(0, Math.min(1, valStep / valTotal));
        const base = Math.max(0.9, fallback);
        return Math.max(base, Math.min(1, base + 0.1 * valFrac));
    }
    const epoch = Number.isFinite(last.epoch) ? Number(last.epoch) : null;
    const totalEpochs = Number.isFinite(last.total_epochs) ? Number(last.total_epochs) : null;
    const batch = Number.isFinite(last.batch) ? Number(last.batch) : null;
    // Prefer explicit batches_per_epoch from metrics; otherwise fall back to target_epoch_size if present on job
    const batchesPerEpochMetric = Number.isFinite(last.batches_per_epoch) ? Number(last.batches_per_epoch) : null;
    const batchesPerEpoch =
        batchesPerEpochMetric ||
        (Number.isFinite(job?.config?.scratch?.target_epoch_size) ? Number(job.config.scratch.target_epoch_size) : null);
    if (!epoch || !batch || !batchesPerEpoch || !totalEpochs) {
        return Math.max(0, Math.min(1, fallback));
    }
    const epochIdx0 = Math.max(0, epoch - 1);
    const fracEpoch = Math.max(0, Math.min(1, batch / Math.max(1, batchesPerEpoch)));
    const overall = (epochIdx0 + fracEpoch) / Math.max(1, totalEpochs);
    return Math.max(0, Math.min(1, overall));
}

function computeMetricProgress(job) {
    const metrics = Array.isArray(job?.metrics) ? job.metrics : [];
    if (!metrics.length) return null;
    const last = metrics[metrics.length - 1] || {};
    const batchesPerEpoch = Number.isFinite(last.batches_per_epoch) ? Number(last.batches_per_epoch) : null;
    const totalEpochs = Number.isFinite(last.total_epochs) ? Number(last.total_epochs) : null;
    const batch = Number.isFinite(last.batch) ? Number(last.batch) : null;
    const epoch = Number.isFinite(last.epoch) ? Number(last.epoch) : null;
    if (!batchesPerEpoch || !totalEpochs || !batch || !epoch) return null;
    const done = Math.max(0, (epoch - 1) * batchesPerEpoch + batch);
    const total = Math.max(1, batchesPerEpoch * totalEpochs);
    return Math.max(0, Math.min(1, done / total));
}

function computeMetricEta(job, progressOverride = null) {
    const metrics = Array.isArray(job?.metrics) ? job.metrics : [];
    const created = Number.isFinite(job?.created_at) ? Number(job.created_at) : null;
    const startTs = created;
    const endTs = Date.now() / 1000;
    if (!Number.isFinite(startTs) || endTs <= startTs) return null;
    const progress = progressOverride !== null ? progressOverride : computeMetricProgress(job);
    if (!Number.isFinite(progress) || progress <= 0) return null;
    const elapsed = endTs - startTs;
    const remaining = elapsed * (1 - progress) / progress;
    return remaining > 0 ? remaining : null;
}

function updateSam3Eta(progress) {
    const now = Date.now();
    if (!Number.isFinite(progress) || progress <= 0) {
        sam3EtaState.startTime = sam3EtaState.startTime || now;
        return null;
    }
    if (sam3EtaState.startTime === null) {
        sam3EtaState.startTime = now;
        return null;
    }
    const elapsed = (now - sam3EtaState.startTime) / 1000; // seconds
    if (elapsed <= 0) return null;
    const remaining = elapsed * (1 - progress) / progress;
    return remaining > 0 ? remaining : null;
}

function getMinMax(arr, accessor) {
    let min = Infinity;
    let max = -Infinity;
    arr.forEach((item) => {
        const val = accessor ? accessor(item) : item;
        if (Number.isFinite(val)) {
            if (val < min) min = val;
            if (val > max) max = val;
        }
    });
    if (min === Infinity || max === -Infinity) {
        return [0, 0];
    }
    return [min, max];
}

function drawSam3LossChart() {
    const canvas = sam3TrainElements.lossCanvas;
    const hasAvg = sam3LossState.avgPoints && sam3LossState.avgPoints.length;
    const hasInst = sam3LossState.instPoints && sam3LossState.instPoints.length;
    const hasTrend = sam3LossState.trendPoints && sam3LossState.trendPoints.length;
    if (!canvas || (!hasAvg && !hasInst && !hasTrend)) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = Math.max(canvas.clientWidth || 400, 320);
    const height = Math.max(canvas.clientHeight || 200, 160);
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
    }
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const padding = { top: 14, right: 14, bottom: 18, left: 48 };
    const chartWidth = Math.max(1, width - padding.left - padding.right);
    const chartHeight = Math.max(1, height - padding.top - padding.bottom);

    const seriesX = [];
    if (hasAvg) {
        seriesX.push(sam3LossState.avgPoints[0].x, sam3LossState.avgPoints[sam3LossState.avgPoints.length - 1].x);
    }
    if (hasInst) {
        seriesX.push(sam3LossState.instPoints[0].x, sam3LossState.instPoints[sam3LossState.instPoints.length - 1].x);
    }
    if (hasTrend) {
        seriesX.push(sam3LossState.trendPoints[0].x, sam3LossState.trendPoints[sam3LossState.trendPoints.length - 1].x);
    }
    const minX = Math.min(...seriesX);
    const maxX = Math.max(...seriesX);
    const xRange = Math.max(1, maxX - minX);

    // Shared axis across both series to keep scales comparable
    const allPoints = [];
    if (hasAvg) allPoints.push(...sam3LossState.avgPoints);
    if (hasInst) allPoints.push(...sam3LossState.instPoints);
    if (hasTrend) allPoints.push(...sam3LossState.trendPoints);
    const [yMinRaw, yMaxRaw] = allPoints.length ? getMinMax(allPoints, (p) => p.y) : [0, 1];
    const yMin = Math.max(0, Math.min(yMinRaw, yMaxRaw - 1e-6));
    const yMax = Math.max(yMin + 1e-6, yMaxRaw);
    const yRange = yMax - yMin;

    const tickCount = 4;
    const avgTickStep = yRange / tickCount;
    const avgTicks = [];
    for (let i = 0; i <= tickCount; i += 1) {
        avgTicks.push(yMin + avgTickStep * i);
    }

    // Grid + labels
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px sans-serif";
    avgTicks.forEach((tick) => {
        const norm = (tick - yMin) / yRange;
        const y = padding.top + (1 - norm) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        ctx.fillText(tick.toExponential(1), padding.left - 6, y);
    });

    // Axes
    ctx.strokeStyle = "#94a3b8";
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.stroke();

    // Average loss line (blue)
    if (hasAvg) {
        ctx.strokeStyle = "#2563eb";
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        sam3LossState.avgPoints.forEach((point, idx) => {
            const normX = (point.x - minX) / xRange;
            const normY = (point.y - yMin) / yRange;
            const xPos = padding.left + normX * chartWidth;
            const yPos = padding.top + (1 - normY) * chartHeight;
            if (idx === 0) {
                ctx.moveTo(xPos, yPos);
            } else {
                ctx.lineTo(xPos, yPos);
            }
        });
        ctx.stroke();
    }

    // Instant loss line (orange) on shared axis
    if (hasInst) {
        ctx.strokeStyle = "#f97316";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        sam3LossState.instPoints.forEach((point, idx) => {
            const normX = (point.x - minX) / xRange;
            const normY = (point.y - yMin) / yRange;
            const xPos = padding.left + normX * chartWidth;
            const yPos = padding.top + (1 - normY) * chartHeight;
            if (idx === 0) {
                ctx.moveTo(xPos, yPos);
            } else {
                ctx.lineTo(xPos, yPos);
            }
        });
        ctx.stroke();
    }
    // Trend line (green, dashed)
    if (hasTrend) {
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        sam3LossState.trendPoints.forEach((point, idx) => {
            const normX = (point.x - minX) / xRange;
            const normY = (point.y - yMin) / yRange;
            const xPos = padding.left + normX * chartWidth;
            const yPos = padding.top + (1 - normY) * chartHeight;
            if (idx === 0) {
                ctx.moveTo(xPos, yPos);
            } else {
                ctx.lineTo(xPos, yPos);
            }
        });
        ctx.stroke();
        ctx.setLineDash([]);
    }
    ctx.restore();
}

function updateSam3LossChartFromMetrics(metrics, jobId) {
    if (!sam3TrainElements.lossCanvas) return;
    if (jobId && sam3LossState.jobId !== jobId) {
        resetSam3LossChart(jobId);
        sam3TrainState.valMetrics = [];
    }
    const entries = Array.isArray(metrics) ? metrics : [];
    if (!entries.length) return;
    if (entries.length < sam3LossState.lastMetricCount) {
        resetSam3LossChart(jobId || sam3LossState.jobId);
    }
    const newEntries = entries.slice(sam3LossState.lastMetricCount);
    sam3LossState.lastMetricCount = entries.length;
    if (!newEntries.length) return;
    initSam3LossChart();
    newEntries.forEach((entry) => {
        if (!entry || typeof entry !== "object") return;
        const inst = entry.train_loss_batch !== undefined ? Number(entry.train_loss_batch) : Number(entry.train_loss);
        const avg =
            entry.train_loss_avg10 !== undefined
                ? Number(entry.train_loss_avg10)
                : entry.train_loss_avg !== undefined
                  ? Number(entry.train_loss_avg)
                  : null;
        const stepVal = Number.isFinite(entry.step) ? entry.step : Math.max(sam3LossState.avgPoints.length, sam3LossState.instPoints.length);
        if (Number.isFinite(avg)) {
            sam3LossState.avgPoints.push({ x: stepVal, y: avg });
        }
        if (Number.isFinite(inst)) {
            sam3LossState.instPoints.push({ x: stepVal, y: inst });
        }
        const trendBase = Number.isFinite(avg) ? avg : Number.isFinite(inst) ? inst : null;
        if (trendBase !== null) {
            sam3LossState.rawPoints.push({ x: stepVal, y: trendBase });
            const alpha = sam3TrainState.trendAlpha || 0.05;
            if (!sam3LossState.trendPoints.length) {
                sam3LossState.trendPoints.push({ x: stepVal, y: trendBase });
            } else {
                const prev = sam3LossState.trendPoints[sam3LossState.trendPoints.length - 1].y;
                const smoothed = alpha * trendBase + (1 - alpha) * prev;
                sam3LossState.trendPoints.push({ x: stepVal, y: smoothed });
            }
        }
        if (entry.phase === "val" && (entry.coco_ap !== undefined || entry.coco_ap50 !== undefined)) {
            sam3TrainState.valMetrics.push({
                epoch: Number.isFinite(entry.epoch) ? Number(entry.epoch) : null,
                ap: entry.coco_ap !== undefined ? Number(entry.coco_ap) : null,
                ap50: entry.coco_ap50 !== undefined ? Number(entry.coco_ap50) : null,
                ap75: entry.coco_ap75 !== undefined ? Number(entry.coco_ap75) : null,
                ar10: entry.coco_ar10 !== undefined ? Number(entry.coco_ar10) : null,
                ar100: entry.coco_ar100 !== undefined ? Number(entry.coco_ar100) : null,
            });
        }
    });
    if (sam3LossState.avgPoints.length || sam3LossState.instPoints.length) {
        drawSam3LossChart();
    }
    renderSam3ValMetrics();
}

function recomputeSam3Trend() {
    sam3LossState.trendPoints = [];
    if (!sam3LossState.rawPoints.length) return;
    const alpha = sam3TrainState.trendAlpha || 0.05;
    sam3LossState.rawPoints.forEach((pt, idx) => {
        if (idx === 0) {
            sam3LossState.trendPoints.push({ x: pt.x, y: pt.y });
        } else {
            const prev = sam3LossState.trendPoints[sam3LossState.trendPoints.length - 1].y;
            const smoothed = alpha * pt.y + (1 - alpha) * prev;
            sam3LossState.trendPoints.push({ x: pt.x, y: smoothed });
        }
    });
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

function setSam3Message(text, tone = "info") {
    if (!sam3TrainElements.message) return;
    sam3TrainElements.message.textContent = text || "";
    sam3TrainElements.message.className = `training-message ${tone}`;
}

function updateSam3DatasetSummary(entry) {
    if (!sam3TrainElements.datasetSummary) return;
    if (!entry) {
        sam3TrainElements.datasetSummary.textContent = "Pick a dataset to train.";
        return;
    }
    const coco = entry.coco_ready ? "COCO ready" : "Convert required";
    const src = entry.source || "unknown";
    const counts = [];
    if (entry.image_count) counts.push(`${entry.image_count} images`);
    if (entry.train_count) counts.push(`train ${entry.train_count}`);
    if (entry.val_count) counts.push(`val ${entry.val_count}`);
    const countText = counts.length ? ` • ${counts.join(" / ")}` : "";
    sam3TrainElements.datasetSummary.textContent = `${entry.label || entry.id} (${src}, ${coco})${countText}`;
}

async function loadSam3Datasets() {
    try {
        const resp = await fetch(`${API_ROOT}/sam3/datasets`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        sam3TrainState.datasets = Array.isArray(data) ? data : [];
        if (!sam3TrainState.selectedId && sam3TrainState.datasets.length) {
            sam3TrainState.selectedId = sam3TrainState.datasets[0].id;
        }
        if (sam3TrainElements.datasetSelect) {
            sam3TrainElements.datasetSelect.innerHTML = "";
            sam3TrainState.datasets.forEach((entry) => {
                const opt = document.createElement("option");
                opt.value = entry.id;
                opt.textContent = `${entry.label || entry.id}${entry.coco_ready ? "" : " (needs convert)"}`;
                if (entry.id === sam3TrainState.selectedId) {
                    opt.selected = true;
                }
                sam3TrainElements.datasetSelect.appendChild(opt);
            });
        }
        const selected = sam3TrainState.datasets.find((d) => d.id === sam3TrainState.selectedId) || sam3TrainState.datasets[0];
        sam3TrainState.selectedId = selected ? selected.id : null;
        updateSam3DatasetSummary(selected);
        resetSam3Eta();
    } catch (err) {
        console.error("Failed to load SAM3 datasets", err);
        setSam3Message(`Failed to load datasets: ${err.message || err}`, "error");
    }
}

async function convertSam3Dataset() {
    const datasetId = sam3TrainState.selectedId;
    if (!datasetId) {
        setSam3Message("Select a dataset first.", "warn");
        return;
    }
    setSam3Message("Converting dataset to COCO…", "info");
    try {
        const resp = await fetch(`${API_ROOT}/sam3/datasets/${encodeURIComponent(datasetId)}/convert`, { method: "POST" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const meta = await resp.json();
        setSam3Message("Dataset converted.", "success");
        await loadSam3Datasets();
        return meta;
    } catch (err) {
        console.error("SAM3 convert failed", err);
        setSam3Message(`Convert failed: ${err.message || err}`, "error");
        throw err;
    }
}

// Prompt helper (SAM3) - suggest and score text prompts per class
function setPromptHelperMessage(text, tone = "info") {
    if (!promptHelperElements.message) return;
    promptHelperElements.message.textContent = text || "";
    promptHelperElements.message.className = `training-message ${tone}`;
}

    function setPromptSearchMessage(text, tone = "info") {
        if (!promptSearchElements.message) return;
        promptSearchElements.message.textContent = text || "";
        promptSearchElements.message.className = `training-message ${tone}`;
    }

    function setPromptRecipeMessage(text, tone = "info") {
        if (!promptRecipeElements.message) return;
        promptRecipeElements.message.textContent = text || "";
        promptRecipeElements.message.className = `training-message ${tone}`;
    }

    function setAgentResultsMessage(text, tone = "info") {
        if (!agentElements.results) return;
        const msg = document.createElement("div");
        msg.className = `training-message ${tone}`;
        msg.textContent = text;
        agentElements.results.innerHTML = "";
        agentElements.results.appendChild(msg);
    }

    function renderAgentResults(result) {
        if (!agentElements.results) return;
        agentElements.results.innerHTML = "";
        if (!result || !Array.isArray(result.classes)) {
            const empty = document.createElement("div");
            empty.className = "training-message warn";
            empty.textContent = "No agent mining results yet.";
            agentElements.results.appendChild(empty);
            return;
        }
        const frag = document.createDocumentFragment();
        result.classes.forEach((cls) => {
            const card = document.createElement("div");
            card.className = "training-card";
            const body = document.createElement("div");
            body.className = "training-card__body";
            const recipe = cls.recipe || {};
            const steps = Array.isArray(recipe.steps) ? recipe.steps : [];
            const summary = recipe.summary || {};
            const covPct = Number.isFinite(summary.coverage_rate) ? (summary.coverage_rate * 100).toFixed(1) : "0.0";
            body.innerHTML = `
                <div class="training-history-row">
                    <div class="training-history-title" style="font-size: 20px; font-weight: 700;">${escapeHtml(cls.name || cls.id)}</div>
                    <span class="badge">${steps.length} step${steps.length === 1 ? "" : "s"}</span>
                </div>
                <div class="training-help">GT train/val: ${cls.train_gt || 0}/${cls.val_gt || 0}</div>
                <div><strong>Coverage:</strong> ${summary.covered || 0}/${summary.total_gt || 0} (${covPct}%) • FPs: ${summary.fps || 0}</div>
            `;
            if (steps.length) {
                const table = document.createElement("table");
                table.className = "training-table";
                table.innerHTML = `
                    <thead>
                        <tr><th>#</th><th>Type</th><th>Prompt/Exemplar</th><th>Thr</th><th>Gain</th><th>FPs</th><th>Cov%</th></tr>
                    </thead>
                `;
                const tbody = document.createElement("tbody");
                steps.forEach((step, idx) => {
                    const covAfter = Number.isFinite(step.coverage_after) ? (step.coverage_after * 100).toFixed(1) : "";
                    const row = document.createElement("tr");
                    const label =
                        step.type === "visual" && step.exemplar
                            ? `Exemplar img ${step.exemplar.image_id} bbox ${Array.isArray(step.exemplar.bbox) ? step.exemplar.bbox.join(",") : ""}`
                            : step.prompt || "";
                    row.innerHTML = `
                        <td>${idx + 1}</td>
                        <td>${step.type || "text"}</td>
                        <td>${escapeHtml(label)}</td>
                        <td>${(step.threshold ?? "").toString()}</td>
                        <td>${step.gain ?? ""}</td>
                        <td>${step.fps ?? ""}</td>
                        <td>${covAfter}</td>
                    `;
                    tbody.appendChild(row);
                });
                table.appendChild(tbody);
                body.appendChild(table);
                const meta = cls.meta || {};
                const recap = document.createElement("div");
                recap.className = "training-help";
                const promptsTried = meta.text_prompts ? `${meta.text_prompts} text prompt${meta.text_prompts === 1 ? "" : "s"}` : "text prompt(s)";
                const exemplarsTried = meta.exemplars !== undefined ? `${meta.exemplars} exemplar${meta.exemplars === 1 ? "" : "s"}` : "exemplars";
                const explanation = (cls.recipe && cls.recipe.explanation) || "";
                recap.textContent =
                    explanation ||
                    `Tested ${promptsTried} × ${meta.thresholds || 0} thresholds and ${exemplarsTried} (total candidates: ${meta.total_candidates || steps.length}); best coverage came from the steps above.`;
                body.appendChild(recap);
            } else {
                const empty = document.createElement("div");
                empty.className = "training-help";
                empty.textContent = "No steps proposed for this class.";
                body.appendChild(empty);
            }
            const foot = document.createElement("div");
            foot.className = "training-actions";
            const saveBtn = document.createElement("button");
            saveBtn.type = "button";
            saveBtn.className = "training-button secondary";
            saveBtn.textContent = "Save recipe";
            saveBtn.addEventListener("click", async () => {
                if (!agentElements.datasetSelect) return;
                const datasetId = agentElements.datasetSelect.value;
                const label = prompt(`Recipe label for ${cls.name || cls.id}?`, `${cls.name || cls.id} recipe`);
                if (!label) return;
                try {
                    const resp = await fetch(`${API_ROOT}/agent_mining/recipes`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            dataset_id: datasetId,
                            class_id: cls.id,
                            class_name: cls.name,
                            label,
                            recipe,
                        }),
                    });
                    if (!resp.ok) throw new Error(await resp.text());
                    setAgentStatus(`Saved recipe "${label}".`, "success");
                } catch (err) {
                    console.error("Save recipe failed", err);
                    setAgentStatus(`Save failed: ${err.message || err}`, "error");
                }
            });
            foot.appendChild(saveBtn);
            body.appendChild(foot);
            card.appendChild(body);
            frag.appendChild(card);
        });
        agentElements.results.appendChild(frag);
    }
    function readThresholdList(inputEl, fallback = 0.2) {
        if (!inputEl) return [fallback];
        const raw = inputEl.value || "";
        const parts = raw
            .split(/[,\s]+/)
            .map((s) => s.trim())
            .filter(Boolean)
            .map((s) => parseFloat(s))
            .filter((v) => !Number.isNaN(v) && v >= 0 && v <= 1);
        if (!parts.length) return [fallback];
        const seen = new Set();
        const cleaned = [];
        parts.forEach((v) => {
            const key = v.toFixed(4);
            if (seen.has(key)) return;
            seen.add(key);
            cleaned.push(v);
        });
        return cleaned;
    }

    function parseCsvNumbers(raw, { clampMin = null, clampMax = null } = {}) {
        const parts = String(raw || "")
            .split(/[,\\s]+/)
            .map((p) => p.trim())
            .filter(Boolean);
        const vals = [];
        parts.forEach((p) => {
            const num = parseFloat(p);
            if (!Number.isNaN(num)) {
                let v = num;
                if (clampMin !== null) v = Math.max(clampMin, v);
                if (clampMax !== null) v = Math.min(clampMax, v);
                vals.push(v);
            }
        });
        return vals;
    }

    function setAgentStatus(text, tone = "info") {
        if (!agentElements.status) return;
        agentElements.status.textContent = text || "";
        agentElements.status.className = `training-message ${tone}`;
    }

    function setAgentResultsMessage(text, tone = "info") {
        if (!agentElements.results) return;
        const msg = document.createElement("div");
        msg.className = `training-message ${tone}`;
        msg.textContent = text;
        agentElements.results.innerHTML = "";
        agentElements.results.appendChild(msg);
    }

    function parseCsvNumbers(raw, { clampMin = null, clampMax = null } = {}) {
        const parts = String(raw || "")
            .split(/[,\\s]+/)
            .map((p) => p.trim())
            .filter(Boolean);
        const vals = [];
        parts.forEach((p) => {
            const num = parseFloat(p);
            if (!Number.isNaN(num)) {
                let v = num;
                if (clampMin !== null) v = Math.max(clampMin, v);
                if (clampMax !== null) v = Math.min(clampMax, v);
                vals.push(v);
            }
        });
        return vals;
    }

function updatePromptHelperDatasetSummary(entry) {
    if (!promptHelperElements.datasetSummary) return;
    if (!entry) {
        promptHelperElements.datasetSummary.textContent = "Pick a dataset to score prompts against.";
        return;
    }
    const coco = entry.coco_ready ? "COCO ready" : "Convert required";
    const counts = [];
    if (entry.image_count) counts.push(`${entry.image_count} images`);
    if (entry.train_count) counts.push(`train ${entry.train_count}`);
    if (entry.val_count) counts.push(`val ${entry.val_count}`);
    const countText = counts.length ? ` • ${counts.join(" / ")}` : "";
    promptHelperElements.datasetSummary.textContent = `${entry.label || entry.id} (${coco})${countText}`;
}

async function loadPromptHelperDatasets() {
    try {
        const resp = await fetch(`${API_ROOT}/sam3/datasets`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        promptHelperState.datasets = Array.isArray(data) ? data : [];
        if (!promptHelperState.selectedId && promptHelperState.datasets.length) {
            promptHelperState.selectedId = promptHelperState.datasets[0].id;
        }
        if (promptHelperElements.datasetSelect) {
            promptHelperElements.datasetSelect.innerHTML = "";
            promptHelperState.datasets.forEach((entry) => {
                const opt = document.createElement("option");
                opt.value = entry.id;
                opt.textContent = `${entry.label || entry.id}${entry.coco_ready ? "" : " (needs convert)"}`;
                if (entry.id === promptHelperState.selectedId) {
                    opt.selected = true;
                }
                promptHelperElements.datasetSelect.appendChild(opt);
            });
        }
        const selected = promptHelperState.datasets.find((d) => d.id === promptHelperState.selectedId) || promptHelperState.datasets[0];
        promptHelperState.selectedId = selected ? selected.id : null;
        updatePromptHelperDatasetSummary(selected);
    } catch (err) {
        console.error("Failed to load datasets for prompt helper", err);
        setPromptHelperMessage(`Failed to load datasets: ${err.message || err}`, "error");
    }
}

async function loadPromptHelperPresets() {
    try {
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/presets`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        promptHelperState.presets = Array.isArray(data) ? data : [];
        if (promptHelperElements.presetSelect) {
            promptHelperElements.presetSelect.innerHTML = "";
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "Select preset…";
            promptHelperElements.presetSelect.appendChild(placeholder);
            promptHelperState.presets.forEach((p) => {
                const opt = document.createElement("option");
                opt.value = p.id;
                const ds = p.dataset_id ? ` • ${p.dataset_id}` : "";
                opt.textContent = `${p.label || p.id}${ds}`;
                promptHelperElements.presetSelect.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("Failed to load prompt helper presets", err);
    }
}

async function savePromptHelperPreset() {
    const datasetId = promptHelperState.selectedId;
    if (!datasetId) {
        setPromptHelperMessage("Select a dataset first.", "warn");
        return;
    }
    const promptsMap = collectPromptsFromUi();
    if (!Object.keys(promptsMap).length) {
        setPromptHelperMessage("Add prompts before saving.", "warn");
        return;
    }
    const label = promptHelperElements.presetName?.value?.trim() || "";
    try {
        const form = new FormData();
        form.append("dataset_id", datasetId);
        form.append("label", label);
        form.append("prompts_json", JSON.stringify(promptsMap));
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/presets`, {
            method: "POST",
            body: form,
        });
        if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || `HTTP ${resp.status}`);
        }
        const preset = await resp.json();
        setPromptHelperMessage(`Saved preset ${preset.label || preset.id}.`, "success");
        await loadPromptHelperPresets();
        if (promptHelperElements.presetSelect) {
            promptHelperElements.presetSelect.value = preset.id;
        }
    } catch (err) {
        console.error("Prompt helper preset save failed", err);
        setPromptHelperMessage(`Save failed: ${err.message || err}`, "error");
    }
}

async function loadPromptHelperPresetIntoUi() {
    const presetId = promptHelperElements.presetSelect?.value;
    if (!presetId) {
        setPromptHelperMessage("Choose a preset to load.", "warn");
        return;
    }
    const datasetId = promptHelperState.selectedId;
    if (!datasetId) {
        setPromptHelperMessage("Select a dataset first.", "warn");
        return;
    }
    if (!promptHelperState.suggestions.length) {
        setPromptHelperMessage("Generate prompts first, then load a preset to edit/evaluate.", "warn");
        return;
    }
    try {
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/presets/${encodeURIComponent(presetId)}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const preset = await resp.json();
        if (preset.dataset_id && preset.dataset_id !== datasetId) {
            setPromptHelperMessage(`Preset is for dataset ${preset.dataset_id}; switch dataset to use it.`, "warn");
            return;
        }
        const map = preset.prompts_by_class || {};
        const normalized = {};
        Object.entries(map).forEach(([k, v]) => {
            const vals = Array.isArray(v) ? v.map((s) => String(s).trim()).filter(Boolean) : [];
            if (vals.length) {
                normalized[parseInt(k, 10)] = vals;
            }
        });
        promptHelperState.promptsByClass = normalized;
        renderPromptHelperPrompts();
        setPromptHelperMessage(`Loaded preset ${preset.label || preset.id}.`, "success");
        if (promptHelperElements.evaluateButton) promptHelperElements.evaluateButton.disabled = false;
    } catch (err) {
        console.error("Prompt helper preset load failed", err);
        setPromptHelperMessage(`Load failed: ${err.message || err}`, "error");
    }
}

function formatMetric(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    return Number(value).toFixed(digits);
}

function renderPromptHelperPrompts() {
    if (!promptHelperElements.prompts) return;
    promptHelperElements.prompts.innerHTML = "";
    const classes = promptHelperState.suggestions || [];
    if (!classes.length) {
        const empty = document.createElement("div");
        empty.className = "training-help";
        empty.textContent = "Generate prompts to edit and evaluate.";
        promptHelperElements.prompts.appendChild(empty);
        return;
    }
    const frag = document.createDocumentFragment();
    classes.forEach((cls) => {
        const card = document.createElement("div");
        card.className = "training-card";
        const header = document.createElement("div");
        header.className = "training-card__header";
        const title = document.createElement("div");
        title.className = "training-card__title";
        const metaBits = [];
        if (cls.image_count) metaBits.push(`${cls.image_count} images`);
        if (cls.gt_count) metaBits.push(`${cls.gt_count} boxes`);
        const name = cls.class_name || cls.class_id;
        title.innerHTML = `<strong>${name}</strong>${metaBits.length ? ` <span class="training-help">(${metaBits.join(" / ")})</span>` : ""}`;
        header.appendChild(title);
        card.appendChild(header);
        const body = document.createElement("div");
        body.className = "training-card__body";
        const label = document.createElement("label");
        label.textContent = "Prompts (comma or newline separated)";
        label.setAttribute("for", `promptHelperInput-${cls.class_id}`);
        const textarea = document.createElement("textarea");
        textarea.id = `promptHelperInput-${cls.class_id}`;
        textarea.rows = 2;
        const prompts = promptHelperState.promptsByClass[cls.class_id] || cls.default_prompts || [];
        textarea.value = prompts.join(", ");
        body.appendChild(label);
        body.appendChild(textarea);
        const hint = document.createElement("div");
        hint.className = "training-help";
        hint.textContent = "Edit before evaluation; first prompt is used as-is.";
        body.appendChild(hint);
        card.appendChild(body);
        frag.appendChild(card);
    });
    promptHelperElements.prompts.appendChild(frag);
}

function collectPromptsFromUi() {
    const map = {};
    (promptHelperState.suggestions || []).forEach((cls) => {
        const input = document.getElementById(`promptHelperInput-${cls.class_id}`);
        if (!input) return;
        const raw = input.value || "";
        const parts = raw
            .split(/[\n,]+/)
            .map((p) => p.trim())
            .filter(Boolean);
        if (parts.length) {
            map[cls.class_id] = parts;
        }
    });
    promptHelperState.promptsByClass = map;
    return map;
}

function renderPromptHelperResults(job) {
    if (!promptHelperElements.status || !promptHelperElements.results || !promptHelperElements.summary) return;
    promptHelperElements.status.textContent = `${job.status.toUpperCase()}: ${job.message || ""}`;
    promptHelperElements.summary.textContent = "";
    promptHelperElements.results.innerHTML = "";
    if (promptHelperElements.logs) {
        promptHelperElements.logs.innerHTML = "";
    }
    if (promptHelperElements.summary) {
        promptHelperElements.summary.title =
            "Score = F1 * (0.5 + 0.5 * detection-rate). Higher is better; balances precision, recall, and how many images yielded matches.";
    }
    if (job.error) {
        const errEl = document.createElement("div");
        errEl.className = "training-message error";
        errEl.textContent = job.error;
        promptHelperElements.results.appendChild(errEl);
        return;
    }
    const result = job.result;
    if (!result || !Array.isArray(result.classes)) {
        promptHelperElements.summary.textContent = "No results yet.";
        return;
    }
    const cfg = result.config || {};
    promptHelperElements.summary.textContent = `Dataset ${result.dataset_id || ""} • ${cfg.sample_per_class || "?"} images/class • score ≥ ${cfg.score_threshold ?? "?"} • max dets ${cfg.max_dets ?? "?"} • IoU ${cfg.iou_threshold ?? "?"} • seed ${cfg.seed ?? "?"}`;
    if (Array.isArray(job.logs) && promptHelperElements.logs) {
        const logFrag = document.createDocumentFragment();
        job.logs.slice(-200).forEach((entry) => {
            const div = document.createElement("div");
            div.className = "training-log-line";
            const ts = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : "";
            div.textContent = `${ts ? `[${ts}] ` : ""}${entry.msg || entry.message || entry}`;
            logFrag.appendChild(div);
        });
        promptHelperElements.logs.appendChild(logFrag);
    }
    const frag = document.createDocumentFragment();
    result.classes.forEach((cls) => {
        const card = document.createElement("div");
        card.className = "training-card";
        const header = document.createElement("div");
        header.className = "training-card__header";
        const title = document.createElement("div");
        title.className = "training-card__title";
        title.textContent = `${cls.class_name || cls.class_id} (sampled ${cls.images_sampled || 0} images)`;
        header.appendChild(title);
        card.appendChild(header);
        const body = document.createElement("div");
        body.className = "training-card__body";
        const table = document.createElement("table");
        table.className = "metric-table";
        const thead = document.createElement("thead");
        thead.innerHTML = `
                <tr>
                    <th>Score</th>
                    <th>Prompt</th>
                    <th>Detects/img</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Avg IoU</th>
                    <th>Avg score</th>
                    <th>Preds</th>
                    <th>GTs</th>
                </tr>
            `;
        table.appendChild(thead);
        const tbody = document.createElement("tbody");
        (cls.candidates || []).forEach((cand) => {
            const row = document.createElement("tr");
            const detsPerImg = cls.images_sampled ? (cand.matches || 0) / cls.images_sampled : 0;
            row.innerHTML = `
                    <td>${formatMetric(cand.score, 3)}</td>
                    <td>${cand.prompt}</td>
                    <td>${formatMetric(detsPerImg, 2)}</td>
                    <td>${formatMetric(cand.precision, 3)}</td>
                    <td>${formatMetric(cand.recall, 3)}</td>
                    <td>${formatMetric(cand.avg_iou, 3)}</td>
                    <td>${formatMetric(cand.avg_score, 3)}</td>
                    <td>${cand.preds ?? 0}</td>
                    <td>${cand.gts ?? 0}</td>
                `;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        body.appendChild(table);
        card.appendChild(body);
        frag.appendChild(card);
    });
    promptHelperElements.results.appendChild(frag);
    if (promptHelperElements.applyButton) {
        promptHelperElements.applyButton.disabled = false;
    }
}

function renderPromptSearchResults(job) {
    if (!promptSearchElements.results || !promptSearchElements.status) return;
    promptSearchElements.results.innerHTML = "";
    if (promptSearchElements.logs) promptSearchElements.logs.innerHTML = "";
    if (job.error) {
        const err = document.createElement("div");
        err.className = "training-message error";
        err.textContent = job.error;
        promptSearchElements.results.appendChild(err);
        return;
    }
    const result = job.result;
    if (!result || !Array.isArray(result.classes)) {
        return;
    }
    const cfg = result.config || {};
    const summaryBits = [
        `Dataset ${result.dataset_id || ""}`,
        `${cfg.sample_per_class ?? "?"} pos/imgs`,
        `${cfg.negatives_per_class ?? 0} neg/imgs`,
        `score ≥ ${cfg.score_threshold ?? "?"}`,
        `max dets ${cfg.max_dets ?? "?"}`,
        `IoU ${cfg.iou_threshold ?? "?"}`,
        `precision floor ${cfg.precision_floor ?? "?"}`,
        `seed ${cfg.seed ?? "?"}`,
    ];
    if (cfg.class_id !== undefined && cfg.class_id !== null) {
        summaryBits.unshift(`Class ${cfg.class_id}`);
    }
    if (promptSearchElements.status) {
        promptSearchElements.status.title = "Search score boosts recall/det-rate but penalizes prompts that fall below the precision floor.";
        promptSearchElements.status.textContent = `${job.status.toUpperCase()}: ${job.message || ""}`;
    }
    if (promptSearchElements.message) {
        promptSearchElements.message.textContent = summaryBits.join(" • ");
    }
    const frag = document.createDocumentFragment();
    result.classes.forEach((cls) => {
        const card = document.createElement("div");
        card.className = "training-card";
        const header = document.createElement("div");
        header.className = "training-card__header";
        const title = document.createElement("div");
        title.className = "training-card__title";
        title.textContent = `${cls.class_name || cls.class_id} (pos ${cls.positive_images || 0} / neg ${cls.negative_images || 0})`;
        header.appendChild(title);
        card.appendChild(header);
        const body = document.createElement("div");
        body.className = "training-card__body";
        const table = document.createElement("table");
        table.className = "metric-table";
        const thead = document.createElement("thead");
        thead.innerHTML = `
                <tr>
                    <th>Best?</th>
                    <th>Search score</th>
                    <th>Prompt</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Det rate</th>
                    <th>Avg IoU</th>
                    <th>Preds</th>
                    <th>GTs</th>
                    <th>FPs</th>
                </tr>
            `;
        table.appendChild(thead);
        const tbody = document.createElement("tbody");
        (cls.candidates || []).forEach((cand, idx) => {
            const row = document.createElement("tr");
            if (idx === 0) row.classList.add("metric-table__highlight");
            const detRate = cand.det_rate ?? 0;
            row.innerHTML = `
                    <td>${idx === 0 ? "★" : ""}</td>
                    <td>${formatMetric(cand.search_score, 3)}</td>
                    <td>${cand.prompt}</td>
                    <td>${formatMetric(cand.precision, 3)}</td>
                    <td>${formatMetric(cand.recall, 3)}</td>
                    <td>${formatMetric(detRate, 3)}</td>
                    <td>${formatMetric(cand.avg_iou, 3)}</td>
                    <td>${cand.preds ?? 0}</td>
                    <td>${cand.gts ?? 0}</td>
                    <td>${cand.fps ?? 0}</td>
                `;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        body.appendChild(table);
        card.appendChild(body);
        frag.appendChild(card);
    });
    promptSearchElements.results.appendChild(frag);
}

    function renderPromptRecipeResults(job) {
        if (!promptRecipeElements.results || !promptRecipeElements.status) return;
        promptRecipeElements.results.innerHTML = "";
        if (promptRecipeElements.logs) promptRecipeElements.logs.innerHTML = "";
        if (promptRecipeElements.applyButton) promptRecipeElements.applyButton.disabled = true;
    if (job.error) {
        const err = document.createElement("div");
        err.className = "training-message error";
        err.textContent = job.error;
        promptRecipeElements.results.appendChild(err);
        return;
    }
    const result = job.result;
    if (!result) return;
    const recipe = result.recipe || {};
    const steps = Array.isArray(recipe.steps) ? recipe.steps : [];
    const stepCount = steps.length;
    const summary = recipe.summary || {};
    const posIds = Array.isArray(result.positive_image_ids) ? result.positive_image_ids : [];
    const negIds = Array.isArray(result.negative_image_ids) ? result.negative_image_ids : [];
    const seedVal = result.config ? result.config.seed : null;
    const recipeLabel = result.class_name || result.class_id || "recipe";
    if (promptRecipeElements.status) {
        promptRecipeElements.status.textContent = `${job.status.toUpperCase()}: ${job.message || ""}`;
        promptRecipeElements.status.title =
            "Simulated per-image early stop: run prompts in order, skip covered images to avoid extra FPs, best threshold per prompt, drop zero-gain steps.";
    }
    const frag = document.createDocumentFragment();
    const summaryCard = document.createElement("div");
    summaryCard.className = "training-card";
    const summaryBody = document.createElement("div");
    summaryBody.className = "training-card__body";
    const coverageRate = Number.isFinite(summary.coverage_rate) ? (summary.coverage_rate * 100).toFixed(1) : "0";
    summaryBody.innerHTML = `
        <div><strong>Class:</strong> ${escapeHtml(result.class_name || result.class_id)}</div>
        <div><strong>Recipe:</strong> ${stepCount} step${stepCount === 1 ? "" : "s"} (best threshold per prompt; dropped no-gain steps)</div>
        <div><strong>Simulation:</strong> Per-image early stop; negatives run every step. Precision per step uses only images still active.</div>
        <div><strong>Sample:</strong> ${posIds.length} pos / ${negIds.length} neg${Number.isFinite(seedVal) ? ` (seed ${seedVal})` : ""}</div>
        <div><strong>Coverage:</strong> ${summary.covered || 0}/${summary.total_gt || 0} (${coverageRate}%)
        • FPs: ${summary.fps || 0}
        • Duplicates: ${summary.duplicates || 0}
        • Pos imgs: ${result.positive_images || 0}
        • Neg imgs: ${result.negative_images || 0}</div>
    `;
    if (posIds.length || negIds.length) {
        const samplePreview = document.createElement("div");
        samplePreview.className = "training-help";
        const preview = (arr) => {
            if (!arr.length) return "none";
            const slice = arr.slice(0, 12).join(", ");
            return arr.length > 12 ? `${slice} … (${arr.length} total)` : slice;
        };
        samplePreview.textContent = `Pos IDs: ${preview(posIds)} • Neg IDs: ${preview(negIds)}`;
        summaryBody.appendChild(samplePreview);
        const copyRow = document.createElement("div");
        copyRow.className = "training-actions";
        const makeCopyBtn = (label, ids) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "training-button secondary";
            btn.textContent = label;
            btn.addEventListener("click", async () => {
                const text = ids.join(",");
                try {
                    await navigator.clipboard.writeText(text);
                    setPromptRecipeMessage(`Copied ${label.toLowerCase()}.`, "success");
                } catch (err) {
                    setPromptRecipeMessage(`Copy failed: ${err.message || err}`, "error");
                }
            });
            return btn;
        };
        if (posIds.length) copyRow.appendChild(makeCopyBtn("Copy pos IDs", posIds));
        if (negIds.length) copyRow.appendChild(makeCopyBtn("Copy neg IDs", negIds));
        summaryBody.appendChild(copyRow);
    }
    if (steps.length) {
        const downloadRow = document.createElement("div");
        downloadRow.className = "training-actions";
        const dlBtn = document.createElement("button");
        dlBtn.type = "button";
        dlBtn.className = "training-button secondary";
        dlBtn.textContent = "Download recipe JSON";
        dlBtn.addEventListener("click", () => {
            const payload = {
                id: `recipe_${Date.now()}`,
                label: `${recipeLabel}_recipe`,
                class_name: result.class_name,
                class_id: result.class_id,
                seed: seedVal,
                steps: steps.map((s) => ({ prompt: s.prompt, threshold: s.threshold })),
            };
            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${recipeLabel}_recipe.json`;
            a.click();
            URL.revokeObjectURL(url);
        });
        downloadRow.appendChild(dlBtn);
        summaryBody.appendChild(downloadRow);
    }
    summaryCard.appendChild(summaryBody);
    frag.appendChild(summaryCard);

    if (steps.length) {
        const stepCard = document.createElement("div");
        stepCard.className = "training-card";
        const body = document.createElement("div");
        body.className = "training-card__body";
        const title = document.createElement("div");
        title.className = "training-card__title";
        title.textContent = "Best sequence (run top-to-bottom)";
        body.appendChild(title);
        const helper = document.createElement("div");
        helper.className = "training-help";
        helper.textContent =
            "Run prompts in order. Stops running later prompts on images once their GTs are covered (to reduce FPs). Adds = new GTs covered by this step. Cov% is cumulative. FPs shows step vs running total. Prec is recalculated on images still active at this step.";
        body.appendChild(helper);
        const sequenceLabel = steps
            .map((step) => {
                const thr = typeof step.threshold === "number" ? step.threshold.toFixed(2) : step.threshold;
                const thrLabel = thr !== undefined && thr !== null ? ` @ ${thr}` : "";
                return `${step.prompt || ""}${thrLabel}`.trim();
            })
            .filter(Boolean)
            .join(" → ");
        if (sequenceLabel) {
            const sequence = document.createElement("div");
            sequence.className = "training-help";
            sequence.textContent = `Use this order: ${sequenceLabel}`;
            body.appendChild(sequence);
        }
        const table = document.createElement("table");
        table.className = "metric-table";
        const thead = document.createElement("thead");
        thead.innerHTML = `
            <tr>
                <th>#</th>
                <th>Prompt</th>
                <th>Thr</th>
                <th>Adds</th>
                <th>Cov after %</th>
                <th>FPs (step)</th>
                <th>FPs (total)</th>
                <th>Prec</th>
            </tr>
        `;
        table.appendChild(thead);
        const tbody = document.createElement("tbody");
        steps.forEach((step, idx) => {
            const gain = step.gain ?? 0;
            const gainText = gain > 0 ? `+${gain}` : gain;
            const cov =
                Number.isFinite(step.coverage_after) && step.coverage_after > 0
                    ? (step.coverage_after * 100).toFixed(1)
                    : Number.isFinite(step.cum_coverage)
                    ? (step.cum_coverage * 100).toFixed(1)
                    : "0";
            const stepFps = step.fps ?? 0;
            const totalFps = step.cum_fps ?? stepFps;
            const row = document.createElement("tr");
            if (idx === 0) row.classList.add("metric-table__highlight");
            row.innerHTML = `
                <td>${idx + 1}</td>
                <td>${escapeHtml(step.prompt || "")}</td>
                <td>${step.threshold ?? "–"}</td>
                <td>${gainText}</td>
                <td>${cov}</td>
                <td>${stepFps}</td>
                <td>${totalFps}</td>
                <td>${formatMetric(step.precision, 3)}</td>
            `;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        body.appendChild(table);
        stepCard.appendChild(body);
        frag.appendChild(stepCard);
    }

    const candidates = Array.isArray(result.candidates) ? result.candidates : [];
    if (candidates.length) {
        const candCard = document.createElement("div");
        candCard.className = "training-card";
        const body = document.createElement("div");
        body.className = "training-card__body";
        const helper = document.createElement("div");
        helper.className = "training-help";
        helper.textContent =
            "All tested prompts/thresholds (no early-stop simulation). We already used the best threshold per prompt in the sequence above.";
        body.appendChild(helper);
        const table = document.createElement("table");
        table.className = "metric-table";
        const thead = document.createElement("thead");
        thead.innerHTML = `
            <tr>
                <th>Prompt</th>
                <th>Thr</th>
                <th>Matched GT</th>
                <th>FPs</th>
                <th>Prec</th>
                <th>Rec</th>
                <th>Det rate</th>
                <th>Avg IoU</th>
                <th>Preds</th>
                <th>GTs</th>
            </tr>
        `;
        table.appendChild(thead);
        const tbody = document.createElement("tbody");
        candidates.forEach((cand) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${escapeHtml(cand.prompt || "")}</td>
                <td>${cand.threshold ?? "–"}</td>
                <td>${cand.matched_gt ?? 0}</td>
                <td>${cand.fps ?? 0}</td>
                <td>${formatMetric(cand.precision, 3)}</td>
                <td>${formatMetric(cand.recall, 3)}</td>
                <td>${formatMetric(cand.det_rate, 3)}</td>
                <td>${formatMetric(cand.avg_iou, 3)}</td>
                <td>${cand.preds ?? 0}</td>
                <td>${cand.gts ?? 0}</td>
            `;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = `Show tested prompts (${candidates.length})`;
        details.appendChild(summary);
        details.appendChild(table);
        body.appendChild(details);
        candCard.appendChild(body);
        frag.appendChild(candCard);
    }

    const coverage = Array.isArray(result.coverage_by_image) ? result.coverage_by_image : [];
    if (coverage.length) {
        const covCard = document.createElement("div");
        covCard.className = "training-card";
        const body = document.createElement("div");
        body.className = "training-card__body";
        const helper = document.createElement("div");
        helper.className = "training-help";
        helper.textContent = "Per-image hits on the sampled set (debug view).";
        body.appendChild(helper);
        const list = document.createElement("div");
        coverage.slice(0, 40).forEach((entry) => {
            const hits = (entry.hits || [])
                .map((h) => `#${(h.step ?? 0) + 1}(${h.matched || 0}/${h.fps || 0}fp)`)
                .join(", ");
            const div = document.createElement("div");
            div.className = "training-history-item";
            const kind = entry.type === "neg" ? "NEG" : "POS";
            div.textContent = `[${kind}] ${entry.file_name || entry.image_id}: GT ${entry.gt || 0}, hits [${hits || "none"}]`;
            list.appendChild(div);
        });
        if (coverage.length > 40) {
            const more = document.createElement("div");
            more.className = "training-help";
            more.textContent = `Showing first 40 of ${coverage.length} images.`;
            list.appendChild(more);
        }
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = `Per-image coverage (${coverage.length} images)`;
        details.appendChild(summary);
        details.appendChild(list);
        body.appendChild(details);
        covCard.appendChild(body);
        frag.appendChild(covCard);
    }

    promptRecipeElements.results.appendChild(frag);
    if (promptRecipeElements.logs && Array.isArray(job.logs)) {
        const logFrag = document.createDocumentFragment();
        job.logs.slice(-200).forEach((entry) => {
            const div = document.createElement("div");
            div.className = "training-log-line";
            const ts = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : "";
            div.textContent = `${ts ? `[${ts}] ` : ""}${entry.msg || entry.message || entry}`;
            logFrag.appendChild(div);
        });
        promptRecipeElements.logs.appendChild(logFrag);
    }
    if (promptRecipeElements.applyButton && job.status === "completed") {
        promptRecipeElements.applyButton.disabled = false;
    }
}

async function pollPromptSearchJob(force = false) {
    if (!promptSearchState.activeJobId) return;
    if (promptSearchState.pollHandle && !force) {
        // interval controls timing
    }
    try {
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/jobs/${encodeURIComponent(promptSearchState.activeJobId)}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const job = await resp.json();
        promptSearchState.lastJob = job;
        if (promptSearchElements.logs && Array.isArray(job.logs)) {
            const logFrag = document.createDocumentFragment();
            job.logs.slice(-200).forEach((entry) => {
                const div = document.createElement("div");
                div.className = "training-log-line";
                const ts = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : "";
                div.textContent = `${ts ? `[${ts}] ` : ""}${entry.msg || entry.message || entry}`;
                logFrag.appendChild(div);
            });
            promptSearchElements.logs.innerHTML = "";
            promptSearchElements.logs.appendChild(logFrag);
        }
        if (promptSearchElements.status) {
            const pct = job.progress ? Math.round(job.progress * 100) : 0;
            const steps = job.total_steps ? ` • ${job.completed_steps || 0}/${job.total_steps}` : "";
            promptSearchElements.status.textContent = `${job.status.toUpperCase()}: ${job.message || ""} (${pct}%${steps})`;
        }
        if (job.status === "completed" || job.status === "failed") {
            if (promptSearchState.pollHandle) {
                clearInterval(promptSearchState.pollHandle);
                promptSearchState.pollHandle = null;
            }
            if (promptSearchElements.runButton) promptSearchElements.runButton.disabled = false;
            renderPromptSearchResults(job);
        }
    } catch (err) {
        console.error("Prompt search poll failed", err);
        setPromptSearchMessage(`Poll failed: ${err.message || err}`, "error");
    }
}

async function pollPromptRecipeJob(force = false) {
    if (!promptRecipeState.activeJobId) return;
    if (promptRecipeState.pollHandle && !force) {
        // interval controls timing
    }
    try {
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/jobs/${encodeURIComponent(promptRecipeState.activeJobId)}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const job = await resp.json();
        promptRecipeState.lastJob = job;
        if (promptRecipeElements.logs && Array.isArray(job.logs)) {
            const logFrag = document.createDocumentFragment();
            job.logs.slice(-200).forEach((entry) => {
                const div = document.createElement("div");
                div.className = "training-log-line";
                const ts = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : "";
                div.textContent = `${ts ? `[${ts}] ` : ""}${entry.msg || entry.message || entry}`;
                logFrag.appendChild(div);
            });
            promptRecipeElements.logs.innerHTML = "";
            promptRecipeElements.logs.appendChild(logFrag);
        }
        if (promptRecipeElements.status) {
            const pct = job.progress ? Math.round(job.progress * 100) : 0;
            const steps = job.total_steps ? ` • ${job.completed_steps || 0}/${job.total_steps}` : "";
            promptRecipeElements.status.textContent = `${job.status.toUpperCase()}: ${job.message || ""} (${pct}%${steps})`;
        }
        if (job.status === "completed" || job.status === "failed") {
            if (promptRecipeState.pollHandle) {
                clearInterval(promptRecipeState.pollHandle);
                promptRecipeState.pollHandle = null;
            }
            if (promptRecipeElements.runButton) promptRecipeElements.runButton.disabled = false;
            renderPromptRecipeResults(job);
        }
    } catch (err) {
        console.error("Prompt recipe poll failed", err);
        setPromptRecipeMessage(`Poll failed: ${err.message || err}`, "error");
    }
}

async function startPromptRecipeJob() {
    const datasetId = promptHelperState.selectedId;
    if (!datasetId) {
        setPromptRecipeMessage("Select a dataset first.", "warn");
        return;
    }
    const targetVal = promptRecipeElements.classSelect?.value;
    const classId = targetVal ? parseInt(targetVal, 10) : NaN;
    if (Number.isNaN(classId)) {
        setPromptRecipeMessage("Choose a class to target.", "warn");
        return;
    }
    const promptsMap = collectPromptsFromUi();
    const prompts = promptsMap[classId];
    if (!prompts || !prompts.length) {
        setPromptRecipeMessage("Add prompts for the selected class.", "warn");
        return;
    }
    const sampleSize = readNumberInput(promptRecipeElements.sampleSize, { integer: true }) ?? 30;
    const negatives = readNumberInput(promptRecipeElements.negatives, { integer: true }) ?? 10;
    const maxDets = readNumberInput(promptRecipeElements.maxDets, { integer: true }) ?? 100;
    const iouThreshold = readNumberInput(promptRecipeElements.iouThresh, { integer: false }) ?? 0.5;
    const seed = readNumberInput(promptRecipeElements.seed, { integer: true }) ?? 42;
    const thresholds = readThresholdList(promptRecipeElements.thresholds, 0.2);
    const scoreThreshold = thresholds.length ? thresholds[0] : 0.2;
    const payload = {
        dataset_id: datasetId,
        class_id: classId,
        sample_size: sampleSize,
        negatives,
        max_dets: maxDets,
        iou_threshold: iouThreshold,
        seed,
        score_threshold: scoreThreshold,
        prompts: prompts.map((p) => ({ prompt: p, thresholds })),
    };
    try {
        setPromptRecipeMessage("Starting recipe mining…", "info");
        if (promptRecipeElements.runButton) promptRecipeElements.runButton.disabled = true;
        if (promptRecipeElements.applyButton) promptRecipeElements.applyButton.disabled = true;
        if (promptRecipeElements.status) promptRecipeElements.status.textContent = "Starting recipe job…";
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/recipe`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || `HTTP ${resp.status}`);
        }
        const job = await resp.json();
        promptRecipeState.activeJobId = job.job_id;
        promptRecipeState.lastJob = job;
        if (promptRecipeState.pollHandle) clearInterval(promptRecipeState.pollHandle);
        promptRecipeState.pollHandle = setInterval(() => pollPromptRecipeJob(), 2000);
        pollPromptRecipeJob(true);
    } catch (err) {
        console.error("Prompt recipe start failed", err);
        setPromptRecipeMessage(err.message || "Start failed", "error");
        if (promptRecipeElements.runButton) promptRecipeElements.runButton.disabled = false;
    }
}

async function expandPromptRecipePrompts() {
    const datasetId = promptHelperState.selectedId;
    if (!datasetId) {
        setPromptRecipeMessage("Select a dataset first.", "warn");
        return;
    }
    const targetVal = promptRecipeElements.classSelect?.value;
    const classId = targetVal ? parseInt(targetVal, 10) : NaN;
    if (Number.isNaN(classId)) {
        setPromptRecipeMessage("Choose a class to expand.", "warn");
        return;
    }
    const expandCount = readNumberInput(promptRecipeElements.expandCount, { integer: true }) ?? 10;
    const promptsMap = collectPromptsFromUi();
    const prompts = promptsMap[classId] || [];
    if (!prompts.length) {
        setPromptRecipeMessage("Add at least one prompt for the class, then expand.", "warn");
        return;
    }
    try {
        setPromptRecipeMessage("Requesting Qwen expansions…", "info");
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/expand`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: datasetId,
                class_id: classId,
                base_prompts: prompts,
                max_new: expandCount,
            }),
        });
        if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        const combined = Array.isArray(data.combined) ? data.combined : prompts;
        const target = document.getElementById(`promptHelperInput-${classId}`);
        if (target) {
            target.value = combined.join(", ");
        }
        promptHelperState.promptsByClass[classId] = combined;
        setPromptRecipeMessage(`Added ${Math.max(0, combined.length - prompts.length)} new prompts from Qwen.`, "success");
    } catch (err) {
        console.error("Prompt recipe expand failed", err);
        setPromptRecipeMessage(err.message || "Expand failed", "error");
    }
}

    async function applyLastPromptRecipeToPrompts() {
        const job = promptRecipeState.lastJob;
        const datasetId = promptHelperState.selectedId;
        if (!job || !job.result) {
            setPromptRecipeMessage("Run recipe mining first.", "warn");
            return;
        }
    if (!datasetId) {
        setPromptRecipeMessage("Select a dataset first.", "warn");
        return;
    }
    const result = job.result;
    const classId = result.class_id;
    const steps = result.recipe && Array.isArray(result.recipe.steps) ? result.recipe.steps : [];
    if (classId === undefined || classId === null) {
        setPromptRecipeMessage("No class in the last recipe result.", "warn");
        return;
    }
    if (!steps.length) {
        setPromptRecipeMessage("Recipe is empty; nothing to apply.", "warn");
        return;
    }
    const orderedPrompts = [];
    const seen = new Set();
    steps.forEach((step) => {
        const p = (step.prompt || "").trim();
        if (!p) return;
        const key = p.toLowerCase();
        if (seen.has(key)) return;
        seen.add(key);
        orderedPrompts.push(p);
    });
    if (!orderedPrompts.length) {
        setPromptRecipeMessage("No prompts to apply from recipe.", "warn");
        return;
    }
    const thresholds = [];
    const thrSeen = new Set();
    steps.forEach((step) => {
        if (typeof step.threshold !== "number") return;
        const key = step.threshold.toFixed(4);
        if (thrSeen.has(key)) return;
        thrSeen.add(key);
        thresholds.push(step.threshold);
    });
    promptHelperState.promptsByClass[classId] = orderedPrompts;
    const target = document.getElementById(`promptHelperInput-${classId}`);
    if (target) {
        target.value = orderedPrompts.join(", ");
    }
    if (promptRecipeElements.thresholds && thresholds.length) {
        promptRecipeElements.thresholds.value = thresholds.map((t) => t.toFixed(2)).join(", ");
    }
    setPromptRecipeMessage(
        `Applied recipe to prompts for class ${classId}. Save a preset to keep it.`,
        "success"
    );
}

async function startPromptSearchJob() {
    const datasetId = promptHelperState.selectedId;
    if (!datasetId) {
        setPromptSearchMessage("Select a dataset first.", "warn");
        return;
    }
    if (!promptHelperState.suggestions.length) {
        setPromptSearchMessage("Generate prompts first, then run search.", "warn");
        return;
    }
    const promptsMap = collectPromptsFromUi();
    if (!Object.keys(promptsMap).length) {
        setPromptSearchMessage("Add prompts for at least one class.", "warn");
        return;
    }
    const samplePerClass = readNumberInput(promptSearchElements.sampleSize, { integer: true }) ?? 20;
    const negativesPerClass = readNumberInput(promptSearchElements.negatives, { integer: true }) ?? 20;
    const precisionFloor = readNumberInput(promptSearchElements.precisionFloor, { integer: false }) ?? 0.9;
    const scoreThreshold = readNumberInput(promptSearchElements.scoreThresh, { integer: false }) ?? 0.2;
    const maxDets = readNumberInput(promptSearchElements.maxDets, { integer: true }) ?? 100;
    const iouThreshold = readNumberInput(promptSearchElements.iouThresh, { integer: false }) ?? 0.5;
    const seed = readNumberInput(promptSearchElements.seed, { integer: true }) ?? 42;
    const targetVal = promptSearchElements.classSelect?.value;
    let classId = null;
    if (targetVal && targetVal !== "all") {
        const parsed = parseInt(targetVal, 10);
        if (!Number.isNaN(parsed)) classId = parsed;
    }
    try {
        setPromptSearchMessage("Starting prompt search…", "info");
        if (promptSearchElements.runButton) promptSearchElements.runButton.disabled = true;
        if (promptSearchElements.status) promptSearchElements.status.textContent = "Starting search…";
        const payload = {
            dataset_id: datasetId,
            sample_per_class: samplePerClass,
            negatives_per_class: negativesPerClass,
            score_threshold: scoreThreshold,
            max_dets: maxDets,
            iou_threshold: iouThreshold,
            seed,
            precision_floor: precisionFloor,
            prompts_by_class: promptsMap,
            class_id: classId,
        };
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || `HTTP ${resp.status}`);
        }
        const job = await resp.json();
        promptSearchState.activeJobId = job.job_id;
        promptSearchState.lastJob = job;
        if (promptSearchState.pollHandle) clearInterval(promptSearchState.pollHandle);
        promptSearchState.pollHandle = setInterval(() => pollPromptSearchJob(), 2000);
        pollPromptSearchJob(true);
    } catch (err) {
        console.error("Prompt search start failed", err);
        setPromptSearchMessage(`Start failed: ${err.message || err}`, "error");
        if (promptSearchElements.runButton) promptSearchElements.runButton.disabled = false;
    }
}

async function pollPromptHelperJob(force = false) {
    if (!promptHelperState.activeJobId) return;
    if (promptHelperState.pollHandle && !force) {
        // interval controls timing
    }
    try {
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/jobs/${encodeURIComponent(promptHelperState.activeJobId)}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const job = await resp.json();
        promptHelperState.lastJob = job;
        console.info("[prompt-helper] poll", job);
        if (promptHelperElements.logs && Array.isArray(job.logs)) {
            const logFrag = document.createDocumentFragment();
            job.logs.slice(-200).forEach((entry) => {
                const div = document.createElement("div");
                div.className = "training-log-line";
                const ts = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : "";
                div.textContent = `${ts ? `[${ts}] ` : ""}${entry.msg || entry.message || entry}`;
                logFrag.appendChild(div);
            });
            promptHelperElements.logs.innerHTML = "";
            promptHelperElements.logs.appendChild(logFrag);
        }
        if (promptHelperElements.status) {
            const pct = job.progress ? Math.round(job.progress * 100) : 0;
            const steps = job.total_steps ? ` • ${job.completed_steps || 0}/${job.total_steps}` : "";
            promptHelperElements.status.textContent = `${job.status.toUpperCase()}: ${job.message || ""} (${pct}%${steps})`;
        }
        if (job.status === "completed" || job.status === "failed") {
            if (promptHelperState.pollHandle) {
                clearInterval(promptHelperState.pollHandle);
                promptHelperState.pollHandle = null;
            }
            if (promptHelperElements.evaluateButton) {
                promptHelperElements.evaluateButton.disabled = false;
            }
            renderPromptHelperResults(job);
        }
    } catch (err) {
        console.error("Prompt helper poll failed", err);
        setPromptHelperMessage(`Poll failed: ${err.message || err}`, "error");
    }
}

async function generatePromptHelperPrompts() {
    const datasetId = promptHelperState.selectedId;
    if (!datasetId) {
        setPromptHelperMessage("Select a dataset first.", "warn");
        return;
    }
    const maxSynonyms = readNumberInput(promptHelperElements.maxSynonyms, { integer: true }) ?? 3;
    const useQwen = promptHelperElements.useQwen ? !!promptHelperElements.useQwen.checked : true;
    try {
        setPromptHelperMessage("Generating prompt suggestions…", "info");
        if (promptHelperElements.evaluateButton) promptHelperElements.evaluateButton.disabled = true;
        console.info("[prompt-helper] generating suggestions", { datasetId, maxSynonyms, useQwen });
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/suggest`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: datasetId,
                max_synonyms: maxSynonyms,
                use_qwen: useQwen,
            }),
        });
        if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        console.info("[prompt-helper] suggestions received", data);
        promptHelperState.suggestions = Array.isArray(data.classes) ? data.classes : [];
        promptHelperState.promptsByClass = {};
        promptHelperState.suggestions.forEach((cls) => {
            if (Array.isArray(cls.default_prompts) && cls.default_prompts.length) {
                promptHelperState.promptsByClass[cls.class_id] = cls.default_prompts;
            }
        });
        populatePromptSearchClassSelect();
        promptHelperState.lastJob = null;
        promptHelperState.activeJobId = null;
        renderPromptHelperPrompts();
        if (promptHelperElements.results) promptHelperElements.results.innerHTML = "";
        if (promptHelperElements.summary) promptHelperElements.summary.textContent = "Prompts ready; run evaluation to score them.";
        if (promptHelperElements.status) promptHelperElements.status.textContent = "Generated prompts (not evaluated yet).";
        setPromptHelperMessage("Suggestions ready. Review/edit, then evaluate with SAM3.", "success");
        if (promptHelperElements.evaluateButton) promptHelperElements.evaluateButton.disabled = false;
    } catch (err) {
        console.error("Prompt helper suggest failed", err);
        setPromptHelperMessage(`Generation failed: ${err.message || err}`, "error");
    }
}

async function startPromptHelperJob() {
    const datasetId = promptHelperState.selectedId;
    if (!datasetId) {
        setPromptHelperMessage("Select a dataset first.", "warn");
        return;
    }
    const samplePerClass = readNumberInput(promptHelperElements.sampleSize, { integer: true }) ?? 20;
    const maxSynonyms = readNumberInput(promptHelperElements.maxSynonyms, { integer: true }) ?? 3;
    const scoreThreshold = readNumberInput(promptHelperElements.scoreThresh, { integer: false }) ?? 0.2;
    const maxDets = readNumberInput(promptHelperElements.maxDets, { integer: true }) ?? 100;
    const iouThreshold = readNumberInput(promptHelperElements.iouThresh, { integer: false }) ?? 0.5;
    const seed = readNumberInput(promptHelperElements.seed, { integer: true }) ?? 42;
    const useQwen = promptHelperElements.useQwen ? !!promptHelperElements.useQwen.checked : true;
    const promptsMap = collectPromptsFromUi();
    if (!Object.keys(promptsMap).length) {
        setPromptHelperMessage("Add prompts for at least one class, or generate suggestions first.", "warn");
        return;
    }
    setPromptHelperMessage("Starting prompt helper evaluation…", "info");
    if (promptHelperElements.applyButton) {
        promptHelperElements.applyButton.disabled = true;
    }
    if (promptHelperElements.evaluateButton) {
        promptHelperElements.evaluateButton.disabled = true;
    }
    if (promptHelperElements.status) {
        promptHelperElements.status.textContent = "Starting evaluation…";
    }
    try {
        const payload = {
            dataset_id: datasetId,
            sample_per_class: samplePerClass,
            max_synonyms: maxSynonyms,
            score_threshold: scoreThreshold,
            max_dets: maxDets,
            iou_threshold: iouThreshold,
            seed,
            use_qwen: useQwen,
            prompts_by_class: promptsMap,
        };
        console.info("[prompt-helper] starting evaluation", payload);
        const resp = await fetch(`${API_ROOT}/sam3/prompt_helper/jobs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || `HTTP ${resp.status}`);
        }
        const job = await resp.json();
        promptHelperState.activeJobId = job.job_id;
        promptHelperState.lastJob = job;
        if (promptHelperState.pollHandle) {
            clearInterval(promptHelperState.pollHandle);
        }
        promptHelperState.pollHandle = setInterval(() => pollPromptHelperJob(), 2000);
        pollPromptHelperJob(true);
    } catch (err) {
        console.error("Prompt helper start failed", err);
        setPromptHelperMessage(`Start failed: ${err.message || err}`, "error");
    }
}

function applyPromptHelperMapping() {
    const job = promptHelperState.lastJob;
    if (!job || !job.result || !Array.isArray(job.result.classes)) {
        setPromptHelperMessage("No completed results to apply.", "warn");
        return;
    }
    const lines = [];
    job.result.classes.forEach((cls) => {
        const prompts = (cls.candidates || []).slice(0, 3).map((c) => c.prompt).filter(Boolean);
        if (prompts.length) {
            lines.push(`${cls.class_name || cls.class_id}: ${prompts.join(", ")}`);
        }
    });
    if (!lines.length) {
        setPromptHelperMessage("No prompts to apply.", "warn");
        return;
    }
    if (sam3TrainElements.promptVariants) {
        sam3TrainElements.promptVariants.value = lines.join("\n");
    }
    setPromptHelperMessage("Applied top prompts to SAM3 training form.", "success");
}

async function initPromptHelperUi() {
    if (promptHelperInitialized) {
        return;
    }
    promptHelperInitialized = true;
    promptHelperElements.datasetSelect = document.getElementById("promptHelperDatasetSelect");
    promptHelperElements.datasetRefresh = document.getElementById("promptHelperDatasetRefresh");
    promptHelperElements.datasetSummary = document.getElementById("promptHelperDatasetSummary");
    promptHelperElements.sampleSize = document.getElementById("promptHelperSampleSize");
    promptHelperElements.maxSynonyms = document.getElementById("promptHelperMaxSynonyms");
    promptHelperElements.scoreThresh = document.getElementById("promptHelperScoreThresh");
    promptHelperElements.maxDets = document.getElementById("promptHelperMaxDets");
    promptHelperElements.iouThresh = document.getElementById("promptHelperIouThresh");
    promptHelperElements.seed = document.getElementById("promptHelperSeed");
    promptHelperElements.useQwen = document.getElementById("promptHelperUseQwen");
    promptHelperElements.generateButton = document.getElementById("promptHelperGenerateBtn");
    promptHelperElements.evaluateButton = document.getElementById("promptHelperEvaluateBtn");
    promptHelperElements.presetName = document.getElementById("promptHelperPresetName");
    promptHelperElements.presetSaveBtn = document.getElementById("promptHelperPresetSave");
    promptHelperElements.presetSelect = document.getElementById("promptHelperPresetSelect");
    promptHelperElements.presetLoadBtn = document.getElementById("promptHelperPresetLoad");
    promptHelperElements.status = document.getElementById("promptHelperStatus");
    promptHelperElements.summary = document.getElementById("promptHelperSummary");
    promptHelperElements.prompts = document.getElementById("promptHelperPrompts");
    promptHelperElements.results = document.getElementById("promptHelperResults");
    promptHelperElements.logs = document.getElementById("promptHelperLogs");
    promptHelperElements.message = document.getElementById("promptHelperMessage");
    promptHelperElements.applyButton = document.getElementById("promptHelperApplyBtn");
    populatePromptSearchClassSelect();
    promptSearchElements.sampleSize = document.getElementById("promptSearchSampleSize");
    promptSearchElements.negatives = document.getElementById("promptSearchNegatives");
    promptSearchElements.precisionFloor = document.getElementById("promptSearchPrecisionFloor");
    promptSearchElements.scoreThresh = document.getElementById("promptSearchScoreThresh");
    promptSearchElements.maxDets = document.getElementById("promptSearchMaxDets");
    promptSearchElements.iouThresh = document.getElementById("promptSearchIouThresh");
    promptSearchElements.seed = document.getElementById("promptSearchSeed");
    promptSearchElements.runButton = document.getElementById("promptSearchRunBtn");
    promptSearchElements.classSelect = document.getElementById("promptSearchClassSelect");
    promptSearchElements.status = document.getElementById("promptSearchStatus");
    promptSearchElements.logs = document.getElementById("promptSearchLogs");
    promptSearchElements.results = document.getElementById("promptSearchResults");
    promptSearchElements.message = document.getElementById("promptSearchMessage");
    promptRecipeElements.classSelect = document.getElementById("promptRecipeClassSelect");
    promptRecipeElements.sampleSize = document.getElementById("promptRecipeSampleSize");
    promptRecipeElements.negatives = document.getElementById("promptRecipeNegatives");
    promptRecipeElements.thresholds = document.getElementById("promptRecipeThresholds");
    promptRecipeElements.maxDets = document.getElementById("promptRecipeMaxDets");
    promptRecipeElements.iouThresh = document.getElementById("promptRecipeIouThresh");
    promptRecipeElements.seed = document.getElementById("promptRecipeSeed");
    promptRecipeElements.expandCount = document.getElementById("promptRecipeExpandCount");
    promptRecipeElements.expandButton = document.getElementById("promptRecipeExpandBtn");
    promptRecipeElements.runButton = document.getElementById("promptRecipeRunBtn");
    promptRecipeElements.applyButton = document.getElementById("promptRecipeApplyBtn");
    promptRecipeElements.status = document.getElementById("promptRecipeStatus");
    promptRecipeElements.logs = document.getElementById("promptRecipeLogs");
    promptRecipeElements.results = document.getElementById("promptRecipeResults");
    promptRecipeElements.message = document.getElementById("promptRecipeMessage");
    if (promptRecipeElements.applyButton) {
        promptRecipeElements.applyButton.disabled = true;
    }

    sam3RecipeElements.fileInput = document.getElementById("sam3RecipeFile");
    sam3RecipeElements.applyButton = document.getElementById("sam3RecipeApplyButton");
    sam3RecipeElements.status = document.getElementById("sam3RecipeStatus");
    if (promptHelperElements.evaluateButton) {
        promptHelperElements.evaluateButton.disabled = true;
    }

    if (promptHelperElements.datasetSelect) {
        promptHelperElements.datasetSelect.addEventListener("change", (e) => {
            promptHelperState.selectedId = e.target.value;
            const entry = promptHelperState.datasets.find((d) => d.id === promptHelperState.selectedId);
            updatePromptHelperDatasetSummary(entry);
            promptHelperState.suggestions = [];
            promptHelperState.promptsByClass = {};
            renderPromptHelperPrompts();
            if (promptHelperElements.evaluateButton) promptHelperElements.evaluateButton.disabled = true;
            if (promptHelperElements.results) promptHelperElements.results.innerHTML = "";
            if (promptHelperElements.summary) promptHelperElements.summary.textContent = "";
            if (promptSearchElements.results) promptSearchElements.results.innerHTML = "";
            if (promptSearchElements.logs) promptSearchElements.logs.innerHTML = "";
            if (promptSearchElements.status) promptSearchElements.status.textContent = "Idle";
            setPromptSearchMessage("");
            if (promptRecipeElements.results) promptRecipeElements.results.innerHTML = "";
            if (promptRecipeElements.logs) promptRecipeElements.logs.innerHTML = "";
            if (promptRecipeElements.status) promptRecipeElements.status.textContent = "Idle";
            setPromptRecipeMessage("");
            if (promptRecipeElements.applyButton) promptRecipeElements.applyButton.disabled = true;
            sam3RecipeState.recipe = null;
            if (sam3RecipeElements.applyButton) sam3RecipeElements.applyButton.disabled = true;
            if (sam3RecipeElements.status) sam3RecipeElements.status.textContent = "";
            if (promptRecipeState.pollHandle) {
                clearInterval(promptRecipeState.pollHandle);
                promptRecipeState.pollHandle = null;
            }
            promptRecipeState.activeJobId = null;
            promptRecipeState.lastJob = null;
            if (promptSearchElements.classSelect) {
                promptSearchElements.classSelect.innerHTML = "";
                const allOpt = document.createElement("option");
                allOpt.value = "all";
                allOpt.textContent = "All classes";
                promptSearchElements.classSelect.appendChild(allOpt);
                promptSearchElements.classSelect.disabled = true;
            }
            if (promptRecipeElements.classSelect) {
                promptRecipeElements.classSelect.innerHTML = "";
                promptRecipeElements.classSelect.disabled = true;
            }
            populatePromptSearchClassSelect();
        });
    }
    if (promptHelperElements.datasetRefresh) {
        promptHelperElements.datasetRefresh.addEventListener("click", () => {
            loadPromptHelperDatasets().catch((err) => console.error("Prompt helper dataset refresh failed", err));
        });
    }
    if (promptHelperElements.generateButton) {
        promptHelperElements.generateButton.addEventListener("click", () => {
            generatePromptHelperPrompts().catch((err) => console.error("Prompt helper generate failed", err));
        });
    }
    if (promptHelperElements.evaluateButton) {
        promptHelperElements.evaluateButton.addEventListener("click", () => {
            startPromptHelperJob().catch((err) => console.error("Prompt helper start failed", err));
        });
    }
    if (promptHelperElements.presetSaveBtn) {
        promptHelperElements.presetSaveBtn.addEventListener("click", () => {
            savePromptHelperPreset().catch((err) => console.error("Prompt helper preset save failed", err));
        });
    }
    if (promptHelperElements.presetLoadBtn) {
        promptHelperElements.presetLoadBtn.addEventListener("click", () => {
            loadPromptHelperPresetIntoUi().catch((err) => console.error("Prompt helper preset load failed", err));
        });
    }
    if (promptHelperElements.applyButton) {
        promptHelperElements.applyButton.addEventListener("click", applyPromptHelperMapping);
    }
    if (promptSearchElements.runButton) {
        promptSearchElements.runButton.addEventListener("click", () => {
            startPromptSearchJob().catch((err) => console.error("Prompt search start failed", err));
        });
    }
    if (promptRecipeElements.runButton) {
        promptRecipeElements.runButton.addEventListener("click", () => {
            startPromptRecipeJob().catch((err) => console.error("Prompt recipe start failed", err));
        });
    }
    if (promptRecipeElements.applyButton) {
        promptRecipeElements.applyButton.addEventListener("click", () => {
            applyLastPromptRecipeToPrompts().catch((err) => console.error("Prompt recipe apply failed", err));
        });
    }
    if (promptRecipeElements.expandButton) {
        promptRecipeElements.expandButton.addEventListener("click", () => {
            expandPromptRecipePrompts().catch((err) => console.error("Prompt recipe expand failed", err));
        });
    }
    await loadPromptHelperDatasets();
    await loadPromptHelperPresets();
    setPromptHelperMessage("Generate suggestions, edit prompts, then evaluate with SAM3.", "info");
    setPromptSearchMessage("Use the prompts above, then run a targeted search for the best wording.", "info");
    setPromptRecipeMessage("Mine an ordered recipe per class using the prompts you’ve edited above.", "info");
    // Make sure class dropdowns are visible/enabled when prompts are present.
    populatePromptSearchClassSelect();
}

function populatePromptSearchClassSelect() {
    const classes = promptHelperState.suggestions || [];
    if (promptSearchElements.classSelect) {
        const select = promptSearchElements.classSelect;
        select.innerHTML = "";
        const allOpt = document.createElement("option");
        allOpt.value = "all";
        allOpt.textContent = "All classes";
        select.appendChild(allOpt);
        classes.forEach((cls) => {
            const opt = document.createElement("option");
            opt.value = cls.class_id;
            opt.textContent = cls.class_name || cls.class_id;
            select.appendChild(opt);
        });
        select.value = "all";
        select.disabled = !classes.length;
    }
    if (promptRecipeElements.classSelect) {
        const select = promptRecipeElements.classSelect;
        select.innerHTML = "";
        classes.forEach((cls) => {
            const opt = document.createElement("option");
            opt.value = cls.class_id;
            opt.textContent = cls.class_name || cls.class_id;
            select.appendChild(opt);
        });
        if (classes.length) {
            select.value = classes[0].class_id;
            select.disabled = false;
        } else {
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "No classes yet";
            select.appendChild(placeholder);
            select.value = "";
            select.disabled = true;
        }
    }
}

function renderSam3History(list) {
    if (!sam3TrainElements.history) return;
    sam3TrainElements.history.innerHTML = "";
    if (!Array.isArray(list) || !list.length) {
        const empty = document.createElement("div");
        empty.className = "training-history-item";
        empty.textContent = "No SAM3 training jobs yet.";
        sam3TrainElements.history.appendChild(empty);
        return;
    }
    list.forEach((job) => {
        const item = document.createElement("div");
        item.className = "training-history-item";
        const left = document.createElement("div");
        const created = formatTimestamp(job.created_at || 0);
        left.innerHTML = `<strong>${escapeHtml(job.job_id.slice(0, 8))}</strong><div class="training-help">${escapeHtml(job.status)} • ${escapeHtml(created)}</div>`;
        item.appendChild(left);
        item.addEventListener("click", () => {
            if (job.job_id) {
                pollSam3TrainingJob(job.job_id, { force: true }).catch((err) => console.error("SAM3 poll history failed", err));
            }
        });
        sam3TrainElements.history.appendChild(item);
    });
}

function renderRunStorage(entries, elements) {
    if (!elements.list) return;
    elements.list.innerHTML = "";
    if (!Array.isArray(entries) || !entries.length) {
        const empty = document.createElement("div");
        empty.className = "training-history-item";
        empty.textContent = "No runs found.";
        elements.list.appendChild(empty);
        return;
    }
    entries.forEach((entry) => {
        const item = document.createElement("div");
        item.className = "storage-item";
        const main = document.createElement("div");
        main.className = "storage-main";
        const heading = document.createElement("div");
        heading.innerHTML = `<strong>${escapeHtml(entry.id)}</strong>`;
        if (entry.active) {
            const badge = document.createElement("span");
            badge.className = "storage-badge warn";
            badge.textContent = "Active";
            heading.appendChild(document.createTextNode(" "));
            heading.appendChild(badge);
        }
        if (entry.promoted) {
            const badge = document.createElement("span");
            badge.className = "storage-badge success";
            badge.textContent = "Promoted";
            heading.appendChild(document.createTextNode(" "));
            heading.appendChild(badge);
        }
        main.appendChild(heading);
        const parts = [];
        if (Number.isFinite(entry.size_bytes)) parts.push(`total ${formatBytes(entry.size_bytes)}`);
        if (Number.isFinite(entry.checkpoints_size_bytes) && entry.checkpoints_size_bytes > 0)
            parts.push(`ckpts ${formatBytes(entry.checkpoints_size_bytes)}`);
        if (Number.isFinite(entry.logs_size_bytes) && entry.logs_size_bytes > 0)
            parts.push(`logs ${formatBytes(entry.logs_size_bytes)}`);
        if (Number.isFinite(entry.tensorboard_size_bytes) && entry.tensorboard_size_bytes > 0)
            parts.push(`tensorboard ${formatBytes(entry.tensorboard_size_bytes)}`);
        if (Number.isFinite(entry.dumps_size_bytes) && entry.dumps_size_bytes > 0)
            parts.push(`dumps ${formatBytes(entry.dumps_size_bytes)}`);
        const meta = document.createElement("div");
        meta.className = "storage-meta";
        meta.textContent = parts.length ? parts.join(" • ") : "Empty run folder.";
        main.appendChild(meta);
        const actions = document.createElement("div");
        actions.className = "storage-actions";
        const scopes = [
            { label: "Delete ckpts", scope: "checkpoints" },
            { label: "Delete logs", scope: "logs" },
            { label: "Delete dumps", scope: "dumps" },
            { label: "Delete TB", scope: "tensorboard" },
            { label: "Delete run", scope: "all", danger: true },
        ];
        if (!entry.promoted) {
            scopes.push({ label: "Promote (keep last, strip optimizer)", scope: "promote", danger: false });
        }
        scopes.forEach(({ label, scope, danger }) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.textContent = label;
            btn.className = `training-button${danger ? " training-button-danger" : ""}`;
            btn.disabled = !!entry.active;
            btn.addEventListener("click", () => deleteRunStorage(entry.id, scope));
            actions.appendChild(btn);
        });
        item.appendChild(main);
        item.appendChild(actions);
        elements.list.appendChild(item);
    });
}

async function refreshRunStorage() {
    if (!sam3StorageElements.list) return;
    try {
        const resp = await fetch(`${API_ROOT}/sam3/storage/runs?variant=sam3`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        sam3StorageState.items = Array.isArray(data) ? data : [];
        renderRunStorage(sam3StorageState.items, sam3StorageElements);
    } catch (err) {
        console.error("Failed to load run storage", err);
    }
}

async function deleteRunStorage(runId, scope) {
    if (scope === "promote") {
        const qs = new URLSearchParams({ variant: "sam3" });
        try {
            const resp = await fetch(`${API_ROOT}/sam3/storage/runs/${encodeURIComponent(runId)}/promote?${qs.toString()}`, {
                method: "POST",
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const msg = `Promoted ${runId}: kept ${data.kept || "checkpoint"}, freed ${formatBytes(data.freed_bytes || 0)}.`;
            setSam3Message(msg, "success");
        } catch (err) {
            console.error("Promote run failed", err);
            setSam3Message(`Promote failed: ${err.message || err}`, "error");
        }
    } else {
        const label = scope === "all" ? "entire run folder" : scope;
        let confirmText = `Delete ${label} for ${runId}?`;
        const entry = sam3StorageState.items.find((r) => r.id === runId) || null;
        if (entry && entry.promoted) {
            confirmText = `This run is promoted.\n${confirmText}\nClick OK to delete anyway.`;
            const second = typeof window !== "undefined" ? window.confirm(confirmText) : true;
            if (!second) return;
        } else if (typeof window !== "undefined" && !window.confirm(confirmText)) {
            return;
        }
        const qs = new URLSearchParams({ variant: "sam3", scope });
        try {
            const resp = await fetch(`${API_ROOT}/sam3/storage/runs/${encodeURIComponent(runId)}?${qs.toString()}`, {
                method: "DELETE",
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            setSam3Message(`Deleted ${label} for ${runId}.`, "success");
        } catch (err) {
            console.error("Delete run failed", err);
            setSam3Message(`Delete failed: ${err.message || err}`, "error");
        }
    }
    await refreshRunStorage();
}

async function refreshSam3History() {
    try {
        const resp = await fetch(`${API_ROOT}/sam3/train/jobs`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        renderSam3History(data);
        const latestRunning = Array.isArray(data)
            ? data.find((j) => ["running", "queued", "cancelling"].includes(j.status))
            : null;
        if (latestRunning && latestRunning.job_id) {
            pollSam3TrainingJob(latestRunning.job_id, { force: true, silent: true }).catch((err) =>
                console.error("SAM3 poll history failed", err),
            );
        }
    } catch (err) {
        console.error("Failed to load SAM3 training history", err);
    }
}

function updateSam3Ui(job) {
    if (!job || !sam3TrainElements.statusText) return;
    sam3TrainState.lastJobSnapshot = job;
    const metricProgress = computeMetricProgress(job);
    const progressVal = metricProgress !== null ? metricProgress : computeSam3Progress(job);
    const pctVal = Math.max(0, Math.min(100, progressVal * 100));
    const pct = Number.isFinite(pctVal) ? pctVal : 0;
    const pctText = pct.toFixed(1).replace(/\.0$/, "");
    const lastMetric = job.metrics && job.metrics.length ? job.metrics[job.metrics.length - 1] : null;
    const batch = lastMetric && Number.isFinite(lastMetric.batch) ? lastMetric.batch : null;
    const batchesPerEpoch =
        lastMetric && Number.isFinite(lastMetric.batches_per_epoch) ? lastMetric.batches_per_epoch : null;
    const epoch = lastMetric && Number.isFinite(lastMetric.epoch) ? lastMetric.epoch : null;
    const totalEpochs = lastMetric && Number.isFinite(lastMetric.total_epochs) ? lastMetric.total_epochs : null;
    const lastPhase = lastMetric && lastMetric.phase ? String(lastMetric.phase) : null;
    const valStep = lastPhase === "val" && Number.isFinite(lastMetric?.val_step) ? Number(lastMetric.val_step) : null;
    const valTotal = lastPhase === "val" && Number.isFinite(lastMetric?.val_total) ? Number(lastMetric.val_total) : null;
    let statusText = job.status === "running" || job.status === "queued" ? `Training running, ${pctText}% done` : job.status;
    if (lastPhase === "val" && Number.isFinite(valStep) && Number.isFinite(valTotal)) {
        statusText = `Validation running, ${pctText}% done (batch ${valStep}/${valTotal})`;
    } else if (Number.isFinite(epoch) && Number.isFinite(batch) && Number.isFinite(batchesPerEpoch)) {
        const epochPart = Number.isFinite(totalEpochs) ? `epoch ${epoch}/${totalEpochs}` : `epoch ${epoch}`;
        statusText = job.status === "running" || job.status === "queued" ? `Training running, ${pctText}% done` : job.status;
        statusText += ` (${epochPart}, batch ${batch}/${batchesPerEpoch})`;
    }
    if (job.status === "succeeded") {
        statusText = "Training + validation complete";
    }
    sam3TrainElements.statusText.textContent = statusText;
    if (sam3TrainElements.progressFill) {
        sam3TrainElements.progressFill.style.width = `${pct}%`;
        sam3TrainElements.progressFill.setAttribute("aria-valuenow", pctText);
    }
    if (sam3TrainElements.etaText) {
        const remaining = computeMetricEta(job, metricProgress !== null ? metricProgress : progressVal);
        if (job.status === "succeeded") {
            sam3TrainElements.etaText.textContent = "ETA: complete";
        } else if (remaining !== null && remaining > 0) {
            sam3TrainElements.etaText.textContent = `ETA: ${formatEta(remaining)}`;
        } else {
            sam3TrainElements.etaText.textContent = "ETA: estimating…";
        }
    }
    if (sam3TrainElements.cancelButton) {
        sam3TrainElements.cancelButton.disabled = !job || !["queued", "running", "cancelling"].includes(job.status);
    }
    if (sam3TrainElements.log) {
        const logs = Array.isArray(job.logs) ? job.logs : [];
        const linesDisplay = logs.map((entry) => (entry.message ? entry.message : "")).filter(Boolean).slice(-200);
        sam3TrainElements.log.textContent = linesDisplay.join("\n");
        updateSam3LossChartFromMetrics(job.metrics, job.job_id);
        renderSam3ValMetrics();
    } else {
        // If we lost references (e.g., DOM re-render), rebind and retry
        sam3TrainElements.log = document.getElementById("sam3Log");
        sam3TrainElements.lossCanvas = document.getElementById("sam3LossChart");
        sam3TrainElements.valMetrics = document.getElementById("sam3ValMetrics");
        if (sam3TrainElements.log && sam3TrainElements.lossCanvas && job && Array.isArray(job.logs)) {
            const linesAll = job.logs.map((entry) => (entry.message ? entry.message : "")).filter(Boolean);
            sam3TrainElements.log.textContent = linesAll.slice(-200).join("\n");
            updateSam3LossChartFromMetrics(job.metrics, job.job_id);
            renderSam3ValMetrics();
        }
    }
    if (sam3TrainElements.summary) {
        if (job.result && job.result.checkpoint) {
            const ckpt = escapeHtml(job.result.checkpoint);
            sam3TrainElements.summary.innerHTML = `Checkpoint: <code>${ckpt}</code>`;
            sam3TrainState.latestCheckpoint = job.result.checkpoint;
            sam3TrainElements.summary.style.display = "block";
        } else {
            sam3TrainElements.summary.textContent = "";
            sam3TrainState.latestCheckpoint = null;
            sam3TrainElements.summary.style.display = "none";
        }
    }
    if (sam3TrainElements.balanceSummary) {
        const info = job.result && job.result.balance_info ? String(job.result.balance_info) : "";
        if (info) {
            sam3TrainElements.balanceSummary.textContent = info;
            sam3TrainElements.balanceSummary.style.display = "block";
        } else {
            sam3TrainElements.balanceSummary.textContent = "";
            sam3TrainElements.balanceSummary.style.display = "none";
        }
    }
    if (sam3TrainElements.activateButton) {
        sam3TrainElements.activateButton.disabled = !sam3TrainState.latestCheckpoint;
    }
}

async function pollSam3TrainingJob(jobId, options = {}) {
    if (!jobId) return;
    sam3TrainState.activeJobId = jobId;
    sam3TrainState.lastSeenJob = sam3TrainState.lastSeenJob || {};
    try {
        const resp = await fetch(`${API_ROOT}/sam3/train/jobs/${jobId}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const job = await resp.json();
        updateSam3Ui(job);
        const running = ["queued", "running", "cancelling"].includes(job.status);
        if (running || options.force) {
            if (sam3TrainState.pollHandle) {
                clearTimeout(sam3TrainState.pollHandle);
            }
            sam3TrainState.pollHandle = window.setTimeout(() => {
                pollSam3TrainingJob(jobId).catch((err) => console.error("SAM3 poll failed", err));
            }, 1500);
        } else {
            sam3TrainState.pollHandle = null;
            refreshSam3History();
        }
        sam3TrainState.lastSeenJob[jobId] = job;
    } catch (err) {
        console.error("SAM3 job poll failed", err);
        if (!options.silent) {
            setSam3Message(`Polling failed: ${err.message || err}`, "error");
        }
    }
}

async function startSam3Training() {
    const datasetId = sam3TrainState.selectedId;
    if (!datasetId) {
        setSam3Message("Select a dataset first.", "warn");
        return;
    }
    try {
        await convertSam3Dataset();
    } catch {
        return;
    }
    const payload = { dataset_id: datasetId };
    const maybeNumber = (input) => {
        if (!input || !input.value) return null;
        const num = Number(input.value);
        return Number.isFinite(num) ? num : null;
    };
    const parsePromptVariants = (text) => {
        if (!text) return {};
        const mapping = {};
        text
            .split(/\n+/)
            .map((l) => l.trim())
            .filter(Boolean)
            .forEach((line) => {
                const splitter = line.includes("=") && !line.includes(":") ? "=" : ":";
                const [rawLabel, rest] = line.split(splitter);
                const label = (rawLabel || "").trim();
                if (!label) return;
                const variants = (rest || "")
                    .split(/[,;]/)
                    .map((v) => v.trim())
                    .filter(Boolean);
                if (variants.length) {
                    mapping[label] = variants;
                }
            });
        return mapping;
    };
    if (sam3TrainElements.runName && sam3TrainElements.runName.value.trim()) {
        payload.run_name = sam3TrainElements.runName.value.trim();
    }
    const fields = [
        ["train_batch_size", sam3TrainElements.trainBatch],
        ["val_batch_size", sam3TrainElements.valBatch],
        ["num_train_workers", sam3TrainElements.trainWorkers],
        ["num_val_workers", sam3TrainElements.valWorkers],
        ["max_epochs", sam3TrainElements.epochs],
        ["resolution", sam3TrainElements.resolution],
        ["lr_scale", sam3TrainElements.lrScale],
        ["gradient_accumulation_steps", sam3TrainElements.gradAccum],
        ["val_epoch_freq", sam3TrainElements.valFreq],
        ["scheduler_warmup", sam3TrainElements.warmupSteps],
        ["scheduler_timescale", sam3TrainElements.schedulerTimescale],
    ];
    const capEpoch = sam3TrainElements.capEpoch ? sam3TrainElements.capEpoch.checked : true;
    if (capEpoch) {
        fields.push(["target_epoch_size", sam3TrainElements.targetEpochSize]);
    }
    const capVal = sam3TrainElements.capVal ? sam3TrainElements.capVal.checked : false;
    if (capVal) {
        fields.push(["val_limit", sam3TrainElements.valCapSize]);
    }
    const strategy = sam3TrainElements.balanceStrategy ? sam3TrainElements.balanceStrategy.value : "none";
    if (strategy && strategy !== "none") {
        payload.balance_strategy = strategy;
        payload.balance_classes = true;
        const power = maybeNumber(sam3TrainElements.balancePower);
        const clip = maybeNumber(sam3TrainElements.balanceClip);
        const beta = maybeNumber(sam3TrainElements.balanceBeta);
        const gamma = maybeNumber(sam3TrainElements.balanceGamma);
        if (power !== null && ["inv_sqrt", "clipped_inv"].includes(strategy)) {
            payload.balance_power = power;
        }
        if (clip !== null && strategy === "clipped_inv" && clip >= 1) {
            payload.balance_clip = clip;
        }
        if (beta !== null && strategy === "effective_num") {
            payload.balance_beta = beta;
        }
        if (gamma !== null && strategy === "focal") {
            payload.balance_gamma = gamma;
        }
    } else {
        payload.balance_classes = false;
    }
    fields.forEach(([key, el]) => {
        const val = maybeNumber(el);
        if (val !== null) payload[key] = val;
    });
    if (sam3TrainElements.freezeLanguage && sam3TrainElements.freezeLanguage.checked) {
        payload.freeze_language_backbone = true;
    }
    const langLr = maybeNumber(sam3TrainElements.languageLr);
    if (langLr !== null) {
        payload.language_backbone_lr = langLr;
    }
    if (sam3TrainElements.promptVariants) {
        const variants = parsePromptVariants(sam3TrainElements.promptVariants.value);
        if (Object.keys(variants).length) {
            payload.prompt_variants = variants;
            if (sam3TrainElements.promptRandomize) {
                payload.prompt_randomize = !!sam3TrainElements.promptRandomize.checked;
            }
        }
    }
    if (sam3TrainElements.logAll && sam3TrainElements.logAll.checked) {
        payload.log_every_batch = true;
    }
    const valScore = maybeNumber(sam3TrainElements.valScoreThresh);
    if (valScore !== null) {
        payload.val_score_thresh = valScore;
    }
    const valMaxDets = maybeNumber(sam3TrainElements.valMaxDets);
    if (valMaxDets !== null) {
        payload.val_max_dets = valMaxDets;
    }
    const wantsSegTrain = sam3TrainElements.segTrain ? sam3TrainElements.segTrain.checked : false;
    const bboxOnly = sam3TrainElements.bboxOnly ? sam3TrainElements.bboxOnly.checked : false;
    if (bboxOnly) {
        payload.enable_segmentation_head = false;
        payload.train_segmentation = false;
    } else {
        const segHeadChecked = sam3TrainElements.segHead ? sam3TrainElements.segHead.checked : true;
        payload.enable_segmentation_head = segHeadChecked || wantsSegTrain;
        if (wantsSegTrain) {
            payload.train_segmentation = true;
        }
    }
    setSam3Message("Starting SAM3 training…", "info");
    try {
        const resp = await fetch(`${API_ROOT}/sam3/train/jobs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        sam3TrainState.activeJobId = data.job_id;
        resetSam3LossChart(data.job_id);
        resetSam3Eta();
        pollSam3TrainingJob(data.job_id, { force: true }).catch((err) => console.error("SAM3 poll start failed", err));
        setSam3Message("Job queued.", "success");
        refreshSam3History();
    } catch (err) {
        console.error("SAM3 training start failed", err);
        setSam3Message(err.message || "Failed to start training", "error");
    }
}

async function cancelSam3Training() {
    if (!sam3TrainState.activeJobId) {
        setSam3Message("No active job.", "warn");
        return;
    }
    try {
        const resp = await fetch(`${API_ROOT}/sam3/train/jobs/${sam3TrainState.activeJobId}/cancel`, { method: "POST" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        setSam3Message("Cancellation requested…", "info");
    } catch (err) {
        console.error("SAM3 cancel failed", err);
        setSam3Message(`Cancel failed: ${err.message || err}`, "error");
    }
}

async function activateSam3Checkpoint() {
    const ckpt = sam3TrainState.latestCheckpoint || (sam3TrainState.lastJobSnapshot && sam3TrainState.lastJobSnapshot.result && sam3TrainState.lastJobSnapshot.result.checkpoint);
    if (!ckpt) {
        setSam3Message("No checkpoint to activate.", "warn");
        return;
    }
    const enableSeg = sam3TrainState.lastJobSnapshot && sam3TrainState.lastJobSnapshot.result ? sam3TrainState.lastJobSnapshot.result.enable_segmentation : false;
    const payload = {
        checkpoint_path: ckpt,
        enable_segmentation: enableSeg,
        label: sam3TrainElements.runName && sam3TrainElements.runName.value.trim() ? sam3TrainElements.runName.value.trim() : undefined,
    };
    try {
        const resp = await fetch(`${API_ROOT}/sam3/models/activate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        setSam3Message("Activated SAM3 checkpoint.", "success");
    } catch (err) {
        console.error("SAM3 activate failed", err);
        setSam3Message(`Activate failed: ${err.message || err}`, "error");
    }
}

function setSegBuilderMessage(text, tone = "info") {
    if (!segBuilderElements.message) return;
    segBuilderElements.message.textContent = text || "";
    segBuilderElements.message.className = `training-message ${tone}`;
}

function updateSegBuilderDatasetSummary(entry) {
    if (!segBuilderElements.datasetSummary) return;
    if (!entry) {
        segBuilderElements.datasetSummary.textContent = "Pick a bbox dataset to convert.";
        return;
    }
    const pieces = [];
    pieces.push(entry.type ? `${entry.type.toUpperCase()} dataset` : "bbox dataset");
    if (entry.source) pieces.push(entry.source);
    const counts = [];
    if (entry.image_count) counts.push(`${entry.image_count} images`);
    if (entry.train_count) counts.push(`train ${entry.train_count}`);
    if (entry.val_count) counts.push(`val ${entry.val_count}`);
    if (counts.length) pieces.push(counts.join(" / "));
    segBuilderElements.datasetSummary.textContent = pieces.join(" • ");
}

function renderSegBuilderDatasets(list) {
    segBuilderState.datasets = Array.isArray(list) ? list : [];
    if (!segBuilderElements.datasetSelect) return;
    segBuilderElements.datasetSelect.innerHTML = "";
    segBuilderState.datasets.forEach((entry) => {
        const opt = document.createElement("option");
        opt.value = entry.id;
        const typeLabel = entry.type ? `[${entry.type}] ` : "";
        const cocoNote = entry.coco_ready ? "" : " (needs convert)";
        opt.textContent = `${typeLabel}${entry.label || entry.id}${cocoNote}`;
        if ((entry.type || "bbox") !== "bbox") {
            opt.disabled = true;
        }
        if (entry.id === segBuilderState.selectedId) {
            opt.selected = true;
        }
        segBuilderElements.datasetSelect.appendChild(opt);
    });
    const selected = segBuilderState.datasets.find((d) => d.id === segBuilderState.selectedId && (d.type || "bbox") === "bbox")
        || segBuilderState.datasets.find((d) => (d.type || "bbox") === "bbox")
        || null;
    segBuilderState.selectedId = selected ? selected.id : null;
    if (segBuilderElements.datasetSelect && segBuilderState.selectedId) {
        segBuilderElements.datasetSelect.value = segBuilderState.selectedId;
    }
    updateSegBuilderDatasetSummary(selected);
}

async function refreshSegBuilderDatasets() {
    try {
        const resp = await fetch(`${API_ROOT}/sam3/datasets`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        renderSegBuilderDatasets(data);
    } catch (err) {
        console.error("Failed to load datasets for segmentation builder", err);
        setSegBuilderMessage(`Failed to load datasets: ${err.message || err}`, "error");
    }
}

function renderSegBuilderJobs(list) {
    segBuilderState.jobs = Array.isArray(list) ? list : [];
    const container = segBuilderElements.jobsContainer;
    if (!container) return;
    container.innerHTML = "";
    if (!segBuilderState.jobs.length) {
        const empty = document.createElement("div");
        empty.className = "training-history-item";
        empty.textContent = "No segmentation build jobs yet.";
        container.appendChild(empty);
        return;
    }
    segBuilderState.jobs.forEach((job) => {
        const item = document.createElement("div");
        item.className = "training-history-item";
        const left = document.createElement("div");
        const created = formatTimestamp(job.created_at || 0);
        const planned = job.result && job.result.planned_metadata ? job.result.planned_metadata : job.config && job.config.planned_metadata;
        const targetId = planned && planned.id ? planned.id : "";
        const subtitleParts = [job.status || "unknown", created];
        if (targetId) {
            subtitleParts.unshift(targetId);
        }
        left.innerHTML = `<strong>${escapeHtml(job.job_id ? job.job_id.slice(0, 8) : "job")}</strong><div class="training-help">${escapeHtml(subtitleParts.filter(Boolean).join(" • "))}</div>`;
        item.appendChild(left);
        if (job.message) {
            const msg = document.createElement("div");
            msg.className = "training-help";
            msg.textContent = job.message;
            item.appendChild(msg);
        }
        container.appendChild(item);
    });
}

async function refreshSegBuilderJobs() {
    try {
        const resp = await fetch(`${API_ROOT}/segmentation/build/jobs`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        renderSegBuilderJobs(data);
    } catch (err) {
        console.error("Failed to refresh segmentation jobs", err);
        setSegBuilderMessage(`Failed to refresh jobs: ${err.message || err}`, "error");
    }
}

async function startSegmentationBuild() {
    const datasetId = segBuilderState.selectedId;
    if (!datasetId) {
        setSegBuilderMessage("Select a bbox dataset first.", "warn");
        return;
    }
    const payload = {
        source_dataset_id: datasetId,
        sam_variant: segBuilderElements.samVariant ? segBuilderElements.samVariant.value || "sam3" : "sam3",
    };
    if (segBuilderElements.outputName && segBuilderElements.outputName.value.trim()) {
        payload.output_name = segBuilderElements.outputName.value.trim();
    }
    setSegBuilderMessage("Queuing segmentation build (stub)…", "info");
    try {
        const resp = await fetch(`${API_ROOT}/segmentation/build/jobs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || `HTTP ${resp.status}`);
        }
        const job = await resp.json();
        setSegBuilderMessage("Job queued. Conversion is scaffolded only (no masks yet).", "success");
        await refreshSegBuilderJobs();
    } catch (err) {
        console.error("Segmentation build start failed", err);
        setSegBuilderMessage(err.message || "Failed to start segmentation build", "error");
    }
}

async function initSegBuilderTab() {
    if (segBuilderUiInitialized) return;
    segBuilderUiInitialized = true;
    segBuilderElements.datasetSelect = document.getElementById("segBuilderDatasetSelect");
    segBuilderElements.datasetSummary = document.getElementById("segBuilderDatasetSummary");
    segBuilderElements.outputName = document.getElementById("segBuilderOutputName");
    segBuilderElements.samVariant = document.getElementById("segBuilderVariant");
    segBuilderElements.startButton = document.getElementById("segBuilderStartBtn");
    segBuilderElements.refreshButton = document.getElementById("segBuilderRefreshBtn");
    segBuilderElements.jobsRefresh = document.getElementById("segBuilderJobsRefresh");
    segBuilderElements.jobsContainer = document.getElementById("segBuilderJobs");
    segBuilderElements.message = document.getElementById("segBuilderMessage");
    if (segBuilderElements.datasetSelect) {
        segBuilderElements.datasetSelect.addEventListener("change", () => {
            segBuilderState.selectedId = segBuilderElements.datasetSelect.value || null;
            const entry = segBuilderState.datasets.find((d) => d.id === segBuilderState.selectedId);
            updateSegBuilderDatasetSummary(entry);
        });
    }
    if (segBuilderElements.startButton) {
        segBuilderElements.startButton.addEventListener("click", () => startSegmentationBuild());
    }
    if (segBuilderElements.refreshButton) {
        segBuilderElements.refreshButton.addEventListener("click", () => refreshSegBuilderDatasets());
    }
    if (segBuilderElements.jobsRefresh) {
        segBuilderElements.jobsRefresh.addEventListener("click", () => refreshSegBuilderJobs());
    }
    await refreshSegBuilderDatasets();
    await refreshSegBuilderJobs();
}

async function initSam3TrainUi() {
    if (sam3TrainUiInitialized) return;
    sam3TrainUiInitialized = true;
    sam3TrainState.lastSeenJob = {};
    sam3TrainElements.datasetSelect = document.getElementById("sam3DatasetSelect");
    sam3TrainElements.datasetSummary = document.getElementById("sam3DatasetSummary");
    sam3TrainElements.datasetRefresh = document.getElementById("sam3DatasetRefresh");
    sam3TrainElements.datasetConvert = document.getElementById("sam3DatasetConvert");
    sam3TrainElements.runName = document.getElementById("sam3RunName");
    sam3TrainElements.trainBatch = document.getElementById("sam3TrainBatch");
    sam3TrainElements.valBatch = document.getElementById("sam3ValBatch");
    sam3TrainElements.trainWorkers = document.getElementById("sam3TrainWorkers");
    sam3TrainElements.valWorkers = document.getElementById("sam3ValWorkers");
    sam3TrainElements.epochs = document.getElementById("sam3Epochs");
    sam3TrainElements.resolution = document.getElementById("sam3Resolution");
    sam3TrainElements.lrScale = document.getElementById("sam3LrScale");
    sam3TrainElements.gradAccum = document.getElementById("sam3GradAccum");
    sam3TrainElements.valFreq = document.getElementById("sam3ValFreq");
    sam3TrainElements.capEpoch = document.getElementById("sam3CapEpoch");
    sam3TrainElements.targetEpochSize = document.getElementById("sam3TargetEpochSize");
    sam3TrainElements.capVal = document.getElementById("sam3CapVal");
    sam3TrainElements.valCapSize = document.getElementById("sam3ValCapSize");
    sam3TrainElements.balanceStrategy = document.getElementById("sam3BalanceStrategy");
    sam3TrainElements.balancePower = document.getElementById("sam3BalancePower");
    sam3TrainElements.balanceClip = document.getElementById("sam3BalanceClip");
    sam3TrainElements.balanceBeta = document.getElementById("sam3BalanceBeta");
    sam3TrainElements.balanceGamma = document.getElementById("sam3BalanceGamma");
    sam3TrainElements.balanceDescription = document.getElementById("sam3BalanceDescription");
    sam3TrainElements.warmupSteps = document.getElementById("sam3Warmup");
    sam3TrainElements.schedulerTimescale = document.getElementById("sam3Timescale");
    sam3TrainElements.freezeLanguage = document.getElementById("sam3FreezeLanguage");
    sam3TrainElements.languageLr = document.getElementById("sam3LanguageLr");
    sam3TrainElements.promptVariants = document.getElementById("sam3PromptVariants");
    sam3TrainElements.promptRandomize = document.getElementById("sam3PromptRandomize");
    sam3TrainElements.logAll = document.getElementById("sam3LogAll");
    sam3TrainElements.valScoreThresh = document.getElementById("sam3ValScoreThresh");
    sam3TrainElements.valMaxDets = document.getElementById("sam3ValMaxDets");
    sam3TrainElements.trendSmooth = document.getElementById("sam3TrendSmooth");
    sam3TrainElements.trendSmoothValue = document.getElementById("sam3TrendSmoothValue");
    sam3TrainElements.segHead = document.getElementById("sam3SegHead");
    sam3TrainElements.segTrain = document.getElementById("sam3SegTrain");
    sam3TrainElements.bboxOnly = document.getElementById("sam3BBoxOnly");
    sam3TrainElements.startButton = document.getElementById("sam3StartBtn");
    sam3TrainElements.cancelButton = document.getElementById("sam3CancelBtn");
    sam3TrainElements.statusText = document.getElementById("sam3StatusText");
    sam3TrainElements.etaText = document.getElementById("sam3EtaText");
    sam3TrainElements.progressFill = document.getElementById("sam3ProgressFill");
    sam3TrainElements.message = document.getElementById("sam3Message");
    sam3TrainElements.summary = document.getElementById("sam3Summary");
    sam3TrainElements.balanceSummary = document.getElementById("sam3BalanceSummary");
    sam3TrainElements.log = document.getElementById("sam3Log");
    sam3TrainElements.history = document.getElementById("sam3TrainingHistory");
    sam3TrainElements.lossCanvas = document.getElementById("sam3LossChart");
    sam3TrainElements.valMetrics = document.getElementById("sam3ValMetrics");
    sam3TrainElements.activateButton = document.getElementById("sam3ActivateBtn");
    sam3StorageElements.list = document.getElementById("sam3StorageList");
    sam3StorageElements.refresh = document.getElementById("sam3StorageRefresh");

    if (sam3TrainElements.balanceStrategy) {
        sam3TrainElements.balanceStrategy.addEventListener("change", () => updateBalanceParamVisibility());
        updateBalanceParamVisibility();
    }
    if (sam3TrainElements.trendSmooth && sam3TrainElements.trendSmoothValue) {
        const setTrendLabel = (val) => {
            sam3TrainElements.trendSmoothValue.textContent = Number(val).toFixed(2);
        };
        sam3TrainElements.trendSmooth.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            if (!Number.isFinite(val)) return;
            sam3TrainState.trendAlpha = Math.max(0.001, Math.min(0.9, val));
            setTrendLabel(sam3TrainState.trendAlpha);
            recomputeSam3Trend();
            drawSam3LossChart();
        });
        setTrendLabel(sam3TrainState.trendAlpha || sam3TrainElements.trendSmooth.value || 0.05);
    }
    if (sam3TrainElements.freezeLanguage && sam3TrainElements.languageLr) {
        sam3TrainElements.freezeLanguage.addEventListener("change", () => {
            const frozen = sam3TrainElements.freezeLanguage.checked;
            sam3TrainElements.languageLr.disabled = frozen;
            if (frozen) {
                sam3TrainElements.languageLr.value = "";
            }
        });
    }
    if (sam3TrainElements.segTrain && sam3TrainElements.segHead) {
        sam3TrainElements.segTrain.addEventListener("change", () => {
            if (sam3TrainElements.segTrain.checked) {
                sam3TrainElements.segHead.checked = true;
            }
        });
    }
    if (sam3TrainElements.bboxOnly && sam3TrainElements.segHead && sam3TrainElements.segTrain) {
        sam3TrainElements.bboxOnly.addEventListener("change", () => {
            const bboxMode = sam3TrainElements.bboxOnly.checked;
            if (bboxMode) {
                sam3TrainElements.segHead.checked = false;
                sam3TrainElements.segTrain.checked = false;
            }
        });
    }
    if (sam3TrainElements.datasetSelect) {
        sam3TrainElements.datasetSelect.addEventListener("change", () => {
            sam3TrainState.selectedId = sam3TrainElements.datasetSelect.value;
            const entry = sam3TrainState.datasets.find((d) => d.id === sam3TrainState.selectedId);
            updateSam3DatasetSummary(entry);
            resetSam3Eta();
        });
    }
    if (sam3TrainElements.datasetRefresh) {
        sam3TrainElements.datasetRefresh.addEventListener("click", () => loadSam3Datasets());
    }
    if (sam3TrainElements.capEpoch && sam3TrainElements.targetEpochSize) {
        sam3TrainElements.capEpoch.addEventListener("change", () => {
            sam3TrainElements.targetEpochSize.disabled = !sam3TrainElements.capEpoch.checked;
        });
        sam3TrainElements.targetEpochSize.disabled = !sam3TrainElements.capEpoch.checked;
    }
    if (sam3TrainElements.capVal && sam3TrainElements.valCapSize) {
        sam3TrainElements.capVal.addEventListener("change", () => {
            sam3TrainElements.valCapSize.disabled = !sam3TrainElements.capVal.checked;
        });
        sam3TrainElements.valCapSize.disabled = !sam3TrainElements.capVal.checked;
    }
    if (sam3TrainElements.datasetConvert) {
        sam3TrainElements.datasetConvert.addEventListener("click", () => convertSam3Dataset().catch(() => {}));
    }
    if (sam3TrainElements.startButton) {
        sam3TrainElements.startButton.addEventListener("click", () => startSam3Training());
    }
    if (sam3TrainElements.cancelButton) {
        sam3TrainElements.cancelButton.addEventListener("click", () => cancelSam3Training());
    }
    if (sam3TrainElements.activateButton) {
        sam3TrainElements.activateButton.addEventListener("click", () => activateSam3Checkpoint());
    }
    if (sam3StorageElements.refresh) {
        sam3StorageElements.refresh.addEventListener("click", () => refreshRunStorage());
    }
    await loadSam3Datasets();
    await refreshSam3History();
    await refreshRunStorage();
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
    const maxImageDim = readNumberInput(qwenTrainElements.maxImageDimInput, { integer: true });
    if (maxImageDim !== undefined) {
        const clampedDim = Math.min(Math.max(maxImageDim, 256), 4096);
        payload.max_image_dim = clampedDim;
    }
    const maxDetections = readNumberInput(qwenTrainElements.maxDetectionsInput, { integer: true });
    if (maxDetections !== undefined) {
        const clampedDetections = Math.min(Math.max(maxDetections, 1), 200);
        payload.max_detections_per_sample = clampedDetections;
    }
    if (qwenTrainElements.devicesInput) {
        const rawDevices = (qwenTrainElements.devicesInput.value || "").trim();
        if (rawDevices) {
            const parsed = rawDevices.split(",").map((value) => parseInt(value.trim(), 10)).filter((value) => Number.isFinite(value));
            if (parsed.length) {
                payload.devices = parsed;
            }
        }
    }
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
        const useCachedDataset = useCachedQwenDataset();
        let datasetInfo;
        if (useCachedDataset) {
            const cachedEntry = getSelectedQwenDataset();
            if (!cachedEntry) {
                setQwenTrainMessage("Select a cached dataset or switch back to uploading the current dataset.", "error");
                if (qwenTrainElements.startButton) {
                    qwenTrainElements.startButton.disabled = false;
                }
                return;
            }
            datasetInfo = {
                dataset_root: cachedEntry.dataset_root,
                run_name: qwenTrainElements.runNameInput?.value?.trim() || cachedEntry.id,
            };
        } else {
            datasetInfo = await uploadQwenDatasetStream();
            await loadQwenDatasetList(true);
            if (datasetInfo?.run_name) {
                selectQwenDatasetById(datasetInfo.run_name);
            }
        }
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
    qwenTrainState.lastJobSnapshot = job;
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
    updateQwenEpochDetail(job);
    updateQwenLossChart(job);
}

function findLatestMetric(metrics, predicate) {
    if (!Array.isArray(metrics)) {
        return null;
    }
    for (let index = metrics.length - 1; index >= 0; index -= 1) {
        const entry = metrics[index];
        if (!entry || typeof entry !== "object") {
            continue;
        }
        if (!predicate || predicate(entry)) {
            return entry;
        }
    }
    return null;
}

function updateQwenEpochDetail(job) {
    if (!qwenTrainElements.epochDetail) {
        return;
    }
    const metrics = Array.isArray(job?.metrics) ? job.metrics : [];
    if (!metrics.length) {
        qwenTrainElements.epochDetail.textContent = "Waiting for telemetry…";
        return;
    }
    const latestTrain = findLatestMetric(metrics, (entry) => entry.phase === "train");
    const source = latestTrain || findLatestMetric(metrics);
    if (!source) {
        qwenTrainElements.epochDetail.textContent = "Waiting for telemetry…";
        return;
    }
    const parts = [];
    const epoch = Number.isFinite(source.epoch) ? source.epoch : null;
    const totalEpochs = Number.isFinite(source.total_epochs)
        ? source.total_epochs
        : Number.isFinite(job?.config?.max_epochs)
            ? job.config.max_epochs
            : null;
    if (epoch && totalEpochs) {
        parts.push(`Epoch ${epoch}/${totalEpochs}`);
    } else if (epoch) {
        parts.push(`Epoch ${epoch}`);
    }
    if (source.phase === "train") {
        if (Number.isFinite(source.batch) && Number.isFinite(source.batches_per_epoch)) {
            parts.push(`Batch ${source.batch}/${source.batches_per_epoch}`);
        }
        if (typeof source.epoch_progress === "number") {
            parts.push(`${Math.round(source.epoch_progress * 100)}% of epoch`);
        }
        if (typeof source.train_loss === "number") {
            parts.push(`Loss ${source.train_loss.toFixed(4)}`);
        }
    } else if (source.phase === "val") {
        if (typeof source.value === "number") {
            const metricLabel =
                typeof source.metric === "string" && source.metric.length
                    ? source.metric.replace(/_/g, " ")
                    : "Validation metric";
            parts.push(`${metricLabel} ${source.value.toFixed(4)}`);
        }
    }
    qwenTrainElements.epochDetail.textContent = parts.length ? parts.join(" • ") : "Telemetry updating…";
}

function getQwenLossSeries(job) {
    const metrics = Array.isArray(job?.metrics) ? job.metrics : [];
    const series = [];
    metrics.forEach((entry) => {
        if (!entry || typeof entry !== "object") {
            return;
        }
        if (entry.phase !== "train") {
            return;
        }
        const loss = Number.isFinite(entry.train_loss) ? entry.train_loss : null;
        if (loss === null) {
            return;
        }
        const step =
            Number.isFinite(entry.step) && entry.step !== null
                ? entry.step
                : Number.isFinite(entry.batch)
                    ? entry.batch
                    : series.length + 1;
        series.push({ x: step, y: loss });
    });
    return series;
}

function smoothLossSeries(points, windowSize) {
    if (!Number.isFinite(windowSize) || windowSize <= 1) {
        return points.slice();
    }
    const window = [];
    let sum = 0;
    return points.map((point) => {
        window.push(point.y);
        sum += point.y;
        if (window.length > windowSize) {
            sum -= window.shift();
        }
        const average = sum / window.length;
        return { x: point.x, y: average };
    });
}

function drawQwenLossChart(points) {
    const canvas = qwenTrainElements.lossCanvas;
    if (!canvas) {
        return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        return;
    }
    const width = Math.max(canvas.clientWidth || 400, 320);
    const height = Math.max(canvas.clientHeight || 200, 160);
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
    }
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);
    const axisMinY = 0;
    const axisMaxY = 8;
    const topPadding = 14;
    const bottomPadding = 14;
    const rightPadding = 14;
    const leftPadding = 44;
    const chartWidth = Math.max(1, width - leftPadding - rightPadding);
    const chartHeight = Math.max(1, height - topPadding - bottomPadding);
    const minX = points[0].x;
    const maxX = points[points.length - 1].x;
    const xRange = maxX - minX || 1;
    const yRange = axisMaxY - axisMinY || 1;
    const clampY = (value) => Math.min(Math.max(value, axisMinY), axisMaxY);
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px sans-serif";
    for (let tick = axisMinY; tick <= axisMaxY; tick += 1) {
        const norm = (tick - axisMinY) / yRange;
        const y = topPadding + (1 - norm) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(leftPadding, y);
        ctx.lineTo(width - rightPadding, y);
        ctx.stroke();
        ctx.fillText(String(tick), leftPadding - 6, y);
    }
    ctx.strokeStyle = "#94a3b8";
    ctx.beginPath();
    ctx.moveTo(leftPadding, topPadding);
    ctx.lineTo(leftPadding, topPadding + chartHeight);
    ctx.stroke();
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    points.forEach((point, index) => {
        const normX = (point.x - minX) / xRange;
        const normY = (clampY(point.y) - axisMinY) / yRange;
        const xPos = leftPadding + normX * chartWidth;
        const yPos = topPadding + (1 - normY) * chartHeight;
        if (index === 0) {
            ctx.moveTo(xPos, yPos);
        } else {
            ctx.lineTo(xPos, yPos);
        }
    });
    ctx.stroke();
    ctx.restore();
}

function resetQwenLossCanvas() {
    const canvas = qwenTrainElements.lossCanvas;
    if (!canvas) {
        return;
    }
    const ctx = canvas.getContext("2d");
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width || 0, canvas.height || 0);
    }
}

function updateQwenLossChart(job) {
    if (!qwenTrainElements.lossCanvas || !qwenTrainElements.chartStatus) {
        return;
    }
    const points = getQwenLossSeries(job);
    if (!points.length) {
        resetQwenLossCanvas();
        qwenTrainElements.chartStatus.textContent = "Loss telemetry will appear while a job is running.";
        return;
    }
    const smoothing = Math.max(1, parseInt(qwenTrainState.chartSmoothing, 10) || 1);
    const processed = smoothing > 1 ? smoothLossSeries(points, smoothing) : points;
    drawQwenLossChart(processed);
    const smoothingLabel = smoothing > 1 ? ` • ${smoothing}-point avg` : "";
    qwenTrainElements.chartStatus.textContent = `${points.length} samples${smoothingLabel}`;
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

function scheduleQwenJobPoll(jobId, delayMs = 5000) {
    if (qwenTrainState.pollHandle) {
        clearTimeout(qwenTrainState.pollHandle);
    }
    qwenTrainState.pollHandle = window.setTimeout(() => {
        pollQwenTrainingJob(jobId, { force: true }).catch((error) => console.error("Qwen poll failed", error));
    }, delayMs);
}

async function pollQwenTrainingJob(jobId, { force = false } = {}) {
    if (!jobId) {
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
        const terminalStates = new Set(["succeeded", "failed", "cancelled"]);
        if (!terminalStates.has(job.status)) {
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
        qwenTrainElements.devicesInput = document.getElementById("qwenTrainDevices");
        qwenTrainElements.loraRankInput = document.getElementById("qwenTrainLoraRank");
        qwenTrainElements.loraAlphaInput = document.getElementById("qwenTrainLoraAlpha");
        qwenTrainElements.loraDropoutInput = document.getElementById("qwenTrainLoraDropout");
        qwenTrainElements.patienceInput = document.getElementById("qwenTrainPatience");
        qwenTrainElements.maxImageDimInput = document.getElementById("qwenTrainMaxImageDim");
        qwenTrainElements.maxDetectionsInput = document.getElementById("qwenTrainMaxDetections");
        qwenTrainElements.datasetModeUpload = document.getElementById("qwenDatasetModeUpload");
        qwenTrainElements.datasetModeCached = document.getElementById("qwenDatasetModeCached");
        qwenTrainElements.datasetSelect = document.getElementById("qwenDatasetSelect");
        qwenTrainElements.datasetRefresh = document.getElementById("qwenDatasetRefresh");
        qwenTrainElements.datasetDelete = document.getElementById("qwenDatasetDelete");
        qwenTrainElements.datasetSummary = document.getElementById("qwenDatasetSummary");
        qwenTrainElements.devicesInput = document.getElementById("qwenTrainDevices");
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
        qwenTrainElements.epochDetail = document.getElementById("qwenTrainEpochDetail");
        qwenTrainElements.lossCanvas = document.getElementById("qwenTrainLossCanvas");
        qwenTrainElements.chartStatus = document.getElementById("qwenTrainChartStatus");
        qwenTrainElements.chartSmoothing = document.getElementById("qwenTrainChartSmoothing");
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
        if (qwenTrainElements.datasetModeUpload) {
            qwenTrainElements.datasetModeUpload.addEventListener("change", () => {
                if (qwenTrainElements.datasetModeUpload?.checked) {
                    setQwenDatasetModeState();
                }
            });
        }
        if (qwenTrainElements.datasetModeCached) {
            qwenTrainElements.datasetModeCached.addEventListener("change", () => {
                if (qwenTrainElements.datasetModeCached?.checked) {
                    setQwenDatasetModeState();
                }
            });
        }
        if (qwenTrainElements.datasetSelect) {
            qwenTrainElements.datasetSelect.addEventListener("change", () => {
                qwenDatasetState.selectedId = qwenTrainElements.datasetSelect.value || null;
                updateQwenDatasetSummary();
            });
        }
        if (qwenTrainElements.datasetRefresh) {
            qwenTrainElements.datasetRefresh.addEventListener("click", () => {
                loadQwenDatasetList(true).catch((error) => console.error("Failed to refresh cached datasets", error));
            });
        }
        if (qwenTrainElements.datasetDelete) {
            qwenTrainElements.datasetDelete.addEventListener("click", () => {
                handleQwenDatasetDelete().catch((error) => console.error("Failed to delete cached dataset", error));
            });
        }
        if (qwenTrainElements.chartSmoothing) {
            const initial = parseInt(qwenTrainElements.chartSmoothing.value, 10);
            qwenTrainState.chartSmoothing = Number.isFinite(initial) && initial > 0 ? initial : 1;
            qwenTrainElements.chartSmoothing.addEventListener("change", () => {
                const nextValue = parseInt(qwenTrainElements.chartSmoothing.value, 10);
                qwenTrainState.chartSmoothing = Number.isFinite(nextValue) && nextValue > 0 ? nextValue : 1;
                updateQwenLossChart(qwenTrainState.lastJobSnapshot);
            });
        }
        loadQwenDatasetList().catch((error) => console.error("Failed to load cached datasets", error));
        setQwenDatasetModeState();
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
        tabElements.sam3TrainButton = document.getElementById("tabSam3TrainButton");
        tabElements.agentMiningButton = document.getElementById("tabAgentMiningButton");
        tabElements.promptHelperButton = document.getElementById("tabPromptHelperButton");
        tabElements.sam3PromptModelsButton = document.getElementById("tabSam3PromptModelsButton");
        tabElements.datasetsButton = document.getElementById("tabDatasetsButton");
        tabElements.activeButton = document.getElementById("tabActiveButton");
        tabElements.qwenButton = document.getElementById("tabQwenButton");
        tabElements.predictorsButton = document.getElementById("tabPredictorsButton");
        tabElements.settingsButton = document.getElementById("tabSettingsButton");
        tabElements.labelingPanel = document.getElementById("tabLabeling");
        tabElements.trainingPanel = document.getElementById("tabTraining");
        tabElements.qwenTrainPanel = document.getElementById("tabQwenTrain");
        tabElements.sam3TrainPanel = document.getElementById("tabSam3Train");
        tabElements.agentMiningPanel = document.getElementById("tabAgentMining");
        tabElements.promptHelperPanel = document.getElementById("tabPromptHelper");
        tabElements.sam3PromptModelsPanel = document.getElementById("tabSam3PromptModels");
        tabElements.datasetsPanel = document.getElementById("tabDatasets");
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
        if (tabElements.sam3TrainButton) {
            tabElements.sam3TrainButton.addEventListener("click", () => setActiveTab(TAB_SAM3_TRAIN));
        }
        if (tabElements.agentMiningButton) {
            tabElements.agentMiningButton.addEventListener("click", () => setActiveTab(TAB_AGENT_MINING));
        }
        if (tabElements.promptHelperButton) {
            tabElements.promptHelperButton.addEventListener("click", () => setActiveTab(TAB_PROMPT_HELPER));
        }
        if (tabElements.sam3PromptModelsButton) {
            tabElements.sam3PromptModelsButton.addEventListener("click", () => setActiveTab(TAB_SAM3_PROMPT_MODELS));
        }
        if (tabElements.datasetsButton) {
            tabElements.datasetsButton.addEventListener("click", () => setActiveTab(TAB_DATASETS));
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
        if (tabElements.sam3TrainButton) {
            tabElements.sam3TrainButton.classList.toggle("active", tabName === TAB_SAM3_TRAIN);
        }
        if (tabElements.agentMiningButton) {
            tabElements.agentMiningButton.classList.toggle("active", tabName === TAB_AGENT_MINING);
        }
        if (tabElements.promptHelperButton) {
            tabElements.promptHelperButton.classList.toggle("active", tabName === TAB_PROMPT_HELPER);
        }
        if (tabElements.sam3PromptModelsButton) {
            tabElements.sam3PromptModelsButton.classList.toggle("active", tabName === TAB_SAM3_PROMPT_MODELS);
        }
        if (tabElements.datasetsButton) {
            tabElements.datasetsButton.classList.toggle("active", tabName === TAB_DATASETS);
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
        if (tabElements.sam3TrainPanel) {
            tabElements.sam3TrainPanel.classList.toggle("active", tabName === TAB_SAM3_TRAIN);
        }
        if (tabElements.agentMiningPanel) {
            tabElements.agentMiningPanel.classList.toggle("active", tabName === TAB_AGENT_MINING);
        }
        if (tabElements.promptHelperPanel) {
            tabElements.promptHelperPanel.classList.toggle("active", tabName === TAB_PROMPT_HELPER);
        }
        if (tabElements.sam3PromptModelsPanel) {
            tabElements.sam3PromptModelsPanel.classList.toggle("active", tabName === TAB_SAM3_PROMPT_MODELS);
        }
        if (tabElements.datasetsPanel) {
            tabElements.datasetsPanel.classList.toggle("active", tabName === TAB_DATASETS);
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
        if (tabName === TAB_SAM3_TRAIN && previous !== TAB_SAM3_TRAIN) {
            initSam3TrainUi().catch((err) => console.error("SAM3 UI init failed", err));
        }
        if (tabName === TAB_AGENT_MINING && previous !== TAB_AGENT_MINING) {
            initAgentMiningUi();
        }
        if (tabName === TAB_PROMPT_HELPER && previous !== TAB_PROMPT_HELPER) {
            initPromptHelperUi().catch((err) => console.error("Prompt helper init failed", err));
        }
        if (tabName === TAB_SAM3_PROMPT_MODELS && previous !== TAB_SAM3_PROMPT_MODELS) {
            initSam3PromptModelsUi();
        }
        if (tabName === TAB_DATASETS && previous !== TAB_DATASETS) {
            initDatasetManagerTab().catch((err) => console.error("Dataset manager init failed", err));
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
        backendFuzzerElements.runButton = document.getElementById("runBackendFuzzer");
        backendFuzzerElements.status = document.getElementById("backendFuzzerStatus");
        backendFuzzerElements.log = document.getElementById("backendFuzzerLog");
        backendFuzzerElements.includeQwen = document.getElementById("fuzzerIncludeQwen");
        backendFuzzerElements.includeSam3 = document.getElementById("fuzzerIncludeSam3");
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
        if (backendFuzzerElements.runButton) {
            backendFuzzerElements.runButton.addEventListener("click", () => {
                runBackendFuzzer().catch((err) => console.error("Backend fuzzer failed", err));
            });
        }
    }

    function setSettingsStatus(message, variant = "info") {
        if (!settingsElements.status) {
            return;
        }
        settingsElements.status.textContent = message || "";
        settingsElements.status.className = variant ? `settings-status ${variant}` : "settings-status";
    }

    async function runBackendFuzzer() {
        if (!backendFuzzerElements.runButton || !backendFuzzerElements.status || !backendFuzzerElements.log) {
            return;
        }
        backendFuzzerElements.runButton.disabled = true;
        backendFuzzerElements.status.textContent = "Running fuzzer…";
        backendFuzzerElements.log.textContent = "";
        const includeQwen = Boolean(backendFuzzerElements.includeQwen?.checked);
        const includeSam3 = Boolean(backendFuzzerElements.includeSam3?.checked);
        const tests = [];
        const addLog = (line) => {
            backendFuzzerElements.log.textContent += `${line}\n`;
        };
        const randomImage = () => {
            const canvasEl = document.createElement("canvas");
            canvasEl.width = 96;
            canvasEl.height = 96;
            const ctx = canvasEl.getContext("2d");
            ctx.fillStyle = "#fff";
            ctx.fillRect(0, 0, 96, 96);
            for (let i = 0; i < 20; i++) {
                ctx.fillStyle = `hsl(${Math.random() * 360},80%,60%)`;
                ctx.fillRect(Math.random() * 80, Math.random() * 80, 8 + Math.random() * 8, 8 + Math.random() * 8);
            }
            const dataUrl = canvasEl.toDataURL("image/png");
            return dataUrl.split(",")[1];
        };
        const baseImage = randomImage();
        const addTest = (name, fn) => tests.push({ name, fn });
        addTest("Settings ping", async () => {
            await testApiRootCandidate(API_ROOT);
        });
        addTest("SAM point (sam1)", async () => {
            const payload = {
                point_x: 32,
                point_y: 32,
                image_base64: baseImage,
                sam_variant: "sam1",
            };
            const resp = await fetch(`${API_ROOT}/sam_point`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) throw new Error(await resp.text());
        });
        addTest("SAM point multi (sam1)", async () => {
            const payload = {
                positive_points: [[20, 20], [60, 60]],
                negative_points: [],
                image_base64: baseImage,
                sam_variant: "sam1",
            };
            const resp = await fetch(`${API_ROOT}/sam_point_multi`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) throw new Error(await resp.text());
        });
        if (includeQwen) {
            addTest("Qwen infer (bbox)", async () => {
                const payload = {
                    prompt: "a colorful object",
                    image_base64: baseImage,
                    prompt_type: "bbox",
                    max_results: 3,
                };
                const resp = await fetch(`${API_ROOT}/qwen/infer`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                if (!resp.ok) throw new Error(await resp.text());
            });
        }
        if (includeSam3) {
            addTest("SAM3 text prompt", async () => {
                const payload = {
                    text_prompt: "object",
                    threshold: 0.3,
                    mask_threshold: 0.5,
                    max_results: 5,
                    min_size: 0,
                    simplify_epsilon: 1.0,
                    image_base64: baseImage,
                    sam_variant: "sam3",
                };
                const resp = await fetch(`${API_ROOT}/sam3/text_prompt`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                if (!resp.ok) throw new Error(await resp.text());
            });
        }
        let failures = 0;
        for (const test of tests) {
            addLog(`▶ ${test.name}`);
            try {
                await test.fn();
                addLog(`✔ ${test.name}`);
            } catch (err) {
                failures += 1;
                addLog(`✖ ${test.name}: ${err?.message || err}`);
            }
        }
        backendFuzzerElements.status.textContent = failures === 0 ? "Fuzzer finished: all tests passed" : `Fuzzer finished: ${failures} failed`;
        backendFuzzerElements.status.className = failures === 0 ? "settings-status success" : "settings-status warn";
        backendFuzzerElements.runButton.disabled = false;
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
        initSam3TextUi();
        updateSam3TextButtons();
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

    function initSam3TextUi() {
        if (sam3TextUiInitialized) {
            return;
        }
        sam3TextUiInitialized = true;
        sam3TextElements.panel = document.getElementById("sam3TextPanel");
        sam3TextElements.promptInput = document.getElementById("sam3TextPrompt");
        sam3TextElements.thresholdInput = document.getElementById("sam3Threshold");
        sam3TextElements.maskThresholdInput = document.getElementById("sam3MaskThreshold");
        sam3TextElements.maxResultsInput = document.getElementById("sam3MaxResults");
        sam3TextElements.minSizeInput = document.getElementById("sam3MinSize");
        sam3TextElements.maxPointsInput = document.getElementById("sam3MaxPoints");
        sam3TextElements.epsilonInput = document.getElementById("sam3SimplifyEpsilon");
        sam3TextElements.classSelect = document.getElementById("sam3ClassSelect");
        sam3TextElements.runButton = document.getElementById("sam3RunButton");
        sam3TextElements.autoButton = document.getElementById("sam3RunAutoButton");
        sam3TextElements.similarityButton = document.getElementById("sam3SimilarityButton");
        sam3TextElements.similarityRow = document.getElementById("sam3SimilarityRow");
        sam3TextElements.similarityThresholdInput = document.getElementById("sam3SimilarityThreshold");
        sam3TextElements.status = document.getElementById("sam3TextStatus");
        sam3RecipeElements.fileInput = document.getElementById("sam3RecipeFile");
        sam3RecipeElements.applyButton = document.getElementById("sam3RecipeApplyButton");
        sam3RecipeElements.status = document.getElementById("sam3RecipeStatus");
        sam3RecipeElements.presetSelect = document.getElementById("sam3RecipePresetSelect");
        sam3RecipeElements.presetNameInput = document.getElementById("sam3RecipePresetName");
        sam3RecipeElements.presetSaveButton = document.getElementById("sam3RecipePresetSave");
        sam3RecipeElements.presetLoadButton = document.getElementById("sam3RecipePresetLoad");
        sam3RecipeElements.presetRefreshButton = document.getElementById("sam3RecipePresetRefresh");
        sam3RecipeElements.presetDeleteButton = document.getElementById("sam3RecipePresetDelete");
        if (sam3TextElements.runButton) {
            sam3TextElements.runButton.addEventListener("click", () => handleSam3TextRequest({ auto: false }));
        }
        if (sam3TextElements.autoButton) {
            sam3TextElements.autoButton.addEventListener("click", () => handleSam3TextRequest({ auto: true }));
        }
        if (sam3TextElements.similarityButton) {
            sam3TextElements.similarityButton.addEventListener("click", handleSam3SimilarityPrompt);
        }
        if (sam3RecipeElements.fileInput) {
            sam3RecipeElements.fileInput.addEventListener("change", handleSam3RecipeFile);
        }
        if (sam3RecipeElements.applyButton) {
            sam3RecipeElements.applyButton.addEventListener("click", runSam3RecipeOnImage);
            sam3RecipeElements.applyButton.disabled = true;
        }
        if (sam3RecipeElements.presetSaveButton) {
            sam3RecipeElements.presetSaveButton.addEventListener("click", saveSam3RecipePreset);
        }
        if (sam3RecipeElements.presetLoadButton) {
            sam3RecipeElements.presetLoadButton.addEventListener("click", loadSam3RecipePreset);
        }
        if (sam3RecipeElements.presetRefreshButton) {
            sam3RecipeElements.presetRefreshButton.addEventListener("click", () => {
                loadSam3RecipePresets().catch((err) => console.error("Refresh recipe presets failed", err));
            });
        }
        if (sam3RecipeElements.presetDeleteButton) {
            sam3RecipeElements.presetDeleteButton.addEventListener("click", deleteSam3RecipePreset);
        }
        updateSam3ClassOptions({ resetOverride: true });
        updateSam3TextButtons();
        loadSam3RecipePresets().catch((err) => console.error("Load recipe presets failed", err));
    }

    function setSam3TextStatus(message, variant = "info") {
        const statusEl = sam3TextElements.status;
        if (!statusEl) {
            return;
        }
        statusEl.textContent = message || "";
        statusEl.classList.remove("warn", "error", "success");
        if (!message) {
            return;
        }
        if (variant === "warn" || variant === "error" || variant === "success") {
            statusEl.classList.add(variant);
        }
    }

    function setSam3RecipeStatus(message, variant = "info") {
        const statusEl = sam3RecipeElements.status;
        if (!statusEl) return;
        statusEl.textContent = message || "";
        statusEl.classList.remove("warn", "error", "success");
        if (variant === "warn" || variant === "error" || variant === "success") {
            statusEl.classList.add(variant);
        }
    }

    function parseRecipeJson(text) {
        try {
            const data = JSON.parse(text);
            if (!data || typeof data !== "object") throw new Error("invalid_json");
            const steps = Array.isArray(data.steps) ? data.steps : [];
            const cleanedSteps = steps
                .map((s) => ({
                    prompt: typeof s.prompt === "string" ? s.prompt.trim() : "",
                    threshold:
                        typeof s.threshold === "number" && s.threshold >= 0 && s.threshold <= 1
                            ? s.threshold
                            : null,
                }))
                .filter((s) => s.prompt && s.threshold !== null);
            if (!cleanedSteps.length) throw new Error("no_steps");
            const targetClass = (data.class_name || data.class || data.target_class || "").trim();
            const targetId = data.class_id;
            return {
                label: data.label || data.id || "recipe",
                class_name: targetClass,
                class_id: targetId,
                steps: cleanedSteps,
            };
        } catch (err) {
            throw new Error("parse_failed");
        }
    }

    async function handleSam3RecipeFile(event) {
        const file = event.target?.files?.[0];
        if (!file) {
            return;
        }
        try {
            let recipe = null;
            if (file.name.toLowerCase().endsWith(".zip")) {
                const formData = new FormData();
                formData.append("file", file);
                const resp = await fetch(`${API_ROOT}/agent_mining/recipes/import`, {
                    method: "POST",
                    body: formData,
                });
                if (!resp.ok) throw new Error(await resp.text());
                const data = await resp.json();
                recipe = data.recipe || data;
                recipe.label = data.label || data.id || recipe.label;
                if (!recipe.class_name && data.class_name) recipe.class_name = data.class_name;
                if (recipe.class_id === undefined && data.class_id !== undefined) recipe.class_id = data.class_id;
            } else {
                const text = await file.text();
                recipe = parseRecipeJson(text);
            }
            // Validate class exists in labelmap.
            const classNames = orderedClassNames();
            const targetName = recipe.class_name;
            const lowerToName = new Map(classNames.map((n) => [n.toLowerCase(), n]));
            if (targetName) {
                const found = lowerToName.get(targetName.toLowerCase());
                if (!found) {
                    throw new Error(`class_missing:${targetName}`);
                }
                recipe.class_name = found;
            } else if (typeof recipe.class_id === "number" && classNames[recipe.class_id]) {
                recipe.class_name = classNames[recipe.class_id];
            } else {
                throw new Error("class_missing");
            }
            sam3RecipeState.recipe = recipe;
            if (sam3RecipeElements.applyButton) sam3RecipeElements.applyButton.disabled = false;
            if (sam3RecipeElements.presetNameInput) {
                sam3RecipeElements.presetNameInput.value = recipe.label || recipe.class_name;
            }
            setSam3RecipeStatus(`Loaded recipe for ${recipe.class_name} (${recipe.steps.length} steps).`, "success");
        } catch (err) {
            console.error("Failed to load recipe", err);
            const msg =
                (err && err.message && err.message.startsWith("class_missing"))
                    ? `Class not in label map: ${err.message.split(":")[1] || ""}`
                    : "Invalid recipe file (use zip/json).";
            setSam3RecipeStatus(msg, "error");
            sam3RecipeState.recipe = null;
            if (sam3RecipeElements.applyButton) sam3RecipeElements.applyButton.disabled = true;
        } finally {
            if (sam3RecipeElements.fileInput) sam3RecipeElements.fileInput.value = "";
        }
    }

    async function runSam3RecipeOnImage() {
        const recipe = sam3RecipeState.recipe;
        if (!recipe || !recipe.steps || !recipe.steps.length) {
            setSam3RecipeStatus("Load a recipe JSON first.", "warn");
            return;
        }
        if (!currentImage) {
            setSam3RecipeStatus("Open an image first.", "warn");
            return;
        }
        const classNames = orderedClassNames();
        if (!classNames.includes(recipe.class_name)) {
            setSam3RecipeStatus(`Class ${recipe.class_name} not in current label map.`, "error");
            return;
        }
        if (sam3RecipeElements.applyButton) sam3RecipeElements.applyButton.disabled = true;
        setSam3RecipeStatus(`Running recipe on ${currentImage.name}…`, "info");
        let totalAdded = 0;
        let maskThreshold = parseFloat(sam3TextElements.maskThresholdInput?.value || "0.5");
        if (Number.isNaN(maskThreshold)) {
            maskThreshold = 0.5;
        }
        maskThreshold = Math.min(Math.max(maskThreshold, 0), 1);
        const minSize = Math.max(0, getMinMaskArea());
        const simplifyEps = Math.max(0, getSimplifyEpsilon());
        try {
            for (const step of recipe.steps) {
                const result = await invokeSam3TextPrompt(
                    {
                        text_prompt: step.prompt,
                        threshold: step.threshold,
                        mask_threshold: maskThreshold,
                        max_results: 100,
                        min_size: minSize,
                        simplify_epsilon: simplifyEps,
                    },
                    { auto: false }
                );
                const detections = Array.isArray(result?.detections) ? result.detections : [];
                const added = applySegAwareDetections(detections, recipe.class_name, "SAM3 recipe");
                totalAdded += added;
            }
            setSam3RecipeStatus(`Recipe applied: added ${totalAdded} boxes to ${recipe.class_name}.`, "success");
        } catch (err) {
            console.error("Recipe apply failed", err);
            setSam3RecipeStatus(`Recipe failed: ${err.message || err}`, "error");
        } finally {
            if (sam3RecipeElements.applyButton) sam3RecipeElements.applyButton.disabled = false;
        }
    }

    function refreshSam3SimilarityVisibility() {
        const row = sam3TextElements.similarityRow;
        const btn = sam3TextElements.similarityButton;
        const show = samVariant === "sam3" && samMode;
        if (row) {
            row.style.display = show ? "" : "none";
        }
        if (btn) {
            btn.style.display = show ? "" : "none";
        }
        if (sam3TextElements.similarityThresholdInput) {
            sam3TextElements.similarityThresholdInput.disabled = !show;
        }
    }

    function updateSam3TextButtons() {
        refreshSam3SimilarityVisibility();
        const busy = sam3TextRequestActive || sam3SimilarityRequestActive;
        setButtonDisabled(sam3TextElements.runButton, busy);
        setButtonDisabled(sam3TextElements.autoButton, busy);
        setButtonDisabled(sam3TextElements.similarityButton, busy);
        if (sam3TextElements.runButton) {
            sam3TextElements.runButton.textContent = busy ? "Running…" : "Run SAM3";
        }
        if (sam3TextElements.autoButton) {
            sam3TextElements.autoButton.textContent = busy ? "Running…" : "Run SAM3 + Auto Class";
        }
        if (sam3TextElements.similarityButton) {
            sam3TextElements.similarityButton.textContent = busy ? "Running…" : "SAM3 similarity prompt (use selected box)";
        }
        if (!busy && !(sam3TextElements.status && sam3TextElements.status.textContent)) {
            setSam3TextStatus("Enter a prompt to run SAM3 text segmentation.", "info");
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

    function updateSam3ClassOptions({ resetOverride = false, preserveSelection = false } = {}) {
        if (!sam3TextElements.classSelect) {
            return;
        }
        const classNames = orderedClassNames();
        sam3TextElements.classSelect.innerHTML = "";
        if (!classNames.length) {
            const placeholder = document.createElement("option");
            placeholder.textContent = "Load classes first";
            placeholder.value = "";
            sam3TextElements.classSelect.appendChild(placeholder);
            sam3TextElements.classSelect.disabled = true;
            return;
        }
        sam3TextElements.classSelect.disabled = false;
        const previousValue = preserveSelection ? sam3TextElements.classSelect.value : null;
        classNames.forEach((name) => {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            sam3TextElements.classSelect.appendChild(option);
        });
        let targetValue = null;
        if (preserveSelection && previousValue && classNames.includes(previousValue)) {
            targetValue = previousValue;
        } else if (currentClass && classNames.includes(currentClass)) {
            targetValue = currentClass;
        } else {
            targetValue = classNames[0];
        }
        sam3TextElements.classSelect.value = targetValue;
        if (resetOverride && classNames.length) {
            sam3TextElements.classSelect.value = targetValue;
        }
    }

    function updateQwenRunButton() {
        if (!qwenElements.runButton) {
            return;
        }
        qwenElements.runButton.disabled = !qwenAvailable || qwenRequestActive;
        qwenElements.runButton.textContent = qwenRequestActive ? "Running…" : "Use Qwen";
    }

    function getSam3TargetClass() {
        if (sam3TextElements.classSelect && sam3TextElements.classSelect.value) {
            return sam3TextElements.classSelect.value;
        }
        return currentClass;
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
        const maxDimValue = Number(metadata.max_image_dim);
        const maxDim = Number.isFinite(maxDimValue) && maxDimValue > 0 ? maxDimValue : 1024;
        const detCapValue = Number(metadata.max_detections_per_sample);
        const detectionCap = Number.isFinite(detCapValue) && detCapValue > 0 ? detCapValue : 200;
        qwenModelElements.details.innerHTML = `
            <p><strong>Name:</strong> ${metadata.label || metadata.id || "Custom Run"}</p>
            <p><strong>Base model:</strong> ${metadata.model_id || "Qwen/Qwen2.5-VL-3B-Instruct"}</p>
            <p><strong>Context hint:</strong> ${context}</p>
            <p><strong>Classes:</strong> ${classes}</p>
            <p><strong>Image resize cap:</strong> ${maxDim}px longest side</p>
            <p><strong>Detections/sample:</strong> ${detectionCap} (per-class budget)</p>
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
        if (qwenModelElements.refreshButton) {
            qwenModelElements.refreshButton.disabled = true;
        }
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
        } finally {
            if (qwenModelElements.refreshButton) {
                qwenModelElements.refreshButton.disabled = false;
            }
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
        qwenModelElements.refreshButton = document.getElementById("qwenModelRefreshBtn");
        if (qwenModelElements.refreshButton) {
            qwenModelElements.refreshButton.addEventListener("click", () => {
                refreshQwenModels();
            });
        }
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

    async function handleSam3TextRequest({ auto = false } = {}) {
        if (sam3TextRequestActive) {
            return;
        }
        if (!currentImage || !currentImage.name) {
            setSam3TextStatus("Load an image before running SAM3.", "warn");
            return;
        }
        const prompt = (sam3TextElements.promptInput?.value || "").trim();
        if (!prompt) {
            setSam3TextStatus("Enter a prompt describing what to segment.", "warn");
            sam3TextElements.promptInput?.focus();
            return;
        }
        const targetClass = getSam3TargetClass();
        if (!auto && !targetClass) {
            setSam3TextStatus("Pick a class to assign boxes to before running SAM3.", "warn");
            return;
        }
        let threshold = parseFloat(sam3TextElements.thresholdInput?.value || "0.5");
        if (Number.isNaN(threshold)) {
            threshold = 0.5;
        }
        threshold = Math.min(Math.max(threshold, 0), 1);
        let maskThreshold = parseFloat(sam3TextElements.maskThresholdInput?.value || "0.5");
        if (Number.isNaN(maskThreshold)) {
            maskThreshold = 0.5;
        }
        maskThreshold = Math.min(Math.max(maskThreshold, 0), 1);
        let minSize = parseInt(sam3TextElements.minSizeInput?.value || "0", 10);
        if (Number.isNaN(minSize) || minSize < 0) {
            minSize = 0;
        }
        let maxResults = parseInt(sam3TextElements.maxResultsInput?.value || "20", 10);
        if (Number.isNaN(maxResults)) {
            maxResults = 20;
        }
        maxResults = Math.min(Math.max(maxResults, 1), 100);
        let simplifyEps = parseFloat(sam3TextElements.epsilonInput?.value || "1.0");
        if (Number.isNaN(simplifyEps) || simplifyEps < 0) {
            simplifyEps = 1.0;
        }
        sam3TextRequestActive = true;
        updateSam3TextButtons();
        setSam3TextStatus("Running SAM3…", "info");
        setSamStatus(`Running SAM3 text prompt${auto ? " (auto class)" : ""}…`, { variant: "info", duration: 0 });
        try {
            const result = await invokeSam3TextPrompt(
                {
                    text_prompt: prompt,
                    threshold,
                    mask_threshold: maskThreshold,
                    min_size: minSize,
                    simplify_epsilon: simplifyEps,
                    max_results: maxResults,
                },
                { auto }
            );
        if (currentImage && result?.image_token) {
            rememberSamToken(currentImage.name, samVariant, result.image_token);
        }
        if (Array.isArray(result?.masks) && Array.isArray(result?.detections)) {
            result.detections.forEach((det, idx) => {
                if (det && !det.mask && result.masks[idx]) {
                    det.mask = result.masks[idx];
                }
            });
        }
        if (auto) {
            const added = applySam3AutoDetections(result?.detections || [], targetClass);
            if (added) {
                const shapeLabel = datasetType === "seg" ? "polygon" : "bbox";
                setSam3TextStatus(`SAM3 auto added ${added} ${shapeLabel}${added === 1 ? "" : "es"}.`, "success");
            } else {
                const warning = Array.isArray(result?.warnings) && result.warnings.includes("clip_unavailable")
                    ? "CLIP classifier unavailable; no auto boxes were added."
                    : "SAM3 auto returned no usable boxes.";
                setSam3TextStatus(warning, "warn");
                }
            } else {
                const applied = applySegAwareDetections(result?.detections || [], targetClass, "SAM3");
                if (applied) {
                    const shapeLabel = datasetType === "seg" ? "polygon" : "bbox";
                    setSam3TextStatus(`SAM3 added ${applied} ${shapeLabel}${applied === 1 ? "" : "es"} to ${targetClass}.`, "success");
                } else {
                    const warning = Array.isArray(result?.warnings) && result.warnings.includes("no_results")
                        ? "SAM3 found no matches for that prompt."
                        : "SAM3 returned no usable detections.";
                    setSam3TextStatus(warning, "warn");
                }
            }
        } catch (error) {
            const detail = error?.message || error;
            setSam3TextStatus(`SAM3 error: ${detail}`, "error");
            console.error("SAM3 text prompt failed", error);
        } finally {
            sam3TextRequestActive = false;
            updateSam3TextButtons();
        }
    }

    async function handleSam3SimilarityPrompt() {
        if (sam3SimilarityRequestActive || sam3TextRequestActive) {
            return;
        }
        if (!currentImage || !currentImage.name) {
            setSam3TextStatus("Load an image and select a bbox to use as the exemplar.", "warn");
            return;
        }
        const targetClass = getSam3TargetClass();
        if (!targetClass) {
            setSam3TextStatus("Pick a class to assign detections to before running similarity.", "warn");
            return;
        }
        const exemplar = currentBbox && currentBbox.bbox ? currentBbox.bbox : null;
        if (!exemplar) {
            setSam3TextStatus("Select or draw a bbox to use as the similarity prompt.", "warn");
            return;
        }
        const width = Math.abs(exemplar.width);
        const height = Math.abs(exemplar.height);
        if (width < minBBoxWidth || height < minBBoxHeight) {
            setSam3TextStatus("Exemplar bbox is too small to use for similarity.", "warn");
            return;
        }
        const left = Math.min(exemplar.x, exemplar.x + exemplar.width);
        const top = Math.min(exemplar.y, exemplar.y + exemplar.height);
        let threshold = parseFloat(sam3TextElements.similarityThresholdInput?.value ?? sam3TextElements.thresholdInput?.value ?? "0.5");
        if (Number.isNaN(threshold)) threshold = 0.5;
        threshold = Math.min(Math.max(threshold, 0), 1);
        let maskThreshold = parseFloat(sam3TextElements.maskThresholdInput?.value || "0.5");
        if (Number.isNaN(maskThreshold)) maskThreshold = 0.5;
        maskThreshold = Math.min(Math.max(maskThreshold, 0), 1);
        let minSize = parseInt(sam3TextElements.minSizeInput?.value || "0", 10);
        if (Number.isNaN(minSize) || minSize < 0) minSize = 0;
        let maxResults = parseInt(sam3TextElements.maxResultsInput?.value || "20", 10);
        if (Number.isNaN(maxResults)) maxResults = 20;
        maxResults = Math.min(Math.max(maxResults, 1), 100);
        let simplifyEps = parseFloat(sam3TextElements.epsilonInput?.value || "1.0");
        if (Number.isNaN(simplifyEps) || simplifyEps < 0) simplifyEps = 1.0;

        function yoloBoxToPixelRect(yoloBox) {
            if (!currentImage || !Array.isArray(yoloBox) || yoloBox.length < 4) return null;
            const [cx, cy, wNorm, hNorm] = yoloBox.map(Number);
            const wPx = wNorm * currentImage.width;
            const hPx = hNorm * currentImage.height;
            return {
                x: cx * currentImage.width - wPx / 2,
                y: cy * currentImage.height - hPx / 2,
                width: wPx,
                height: hPx,
            };
        }

        function rectIoU(a, b) {
            const ax2 = a.x + a.width;
            const ay2 = a.y + a.height;
            const bx2 = b.x + b.width;
            const by2 = b.y + b.height;
            const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
            const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
            const inter = ix * iy;
            if (inter <= 0) return 0;
            const union = a.width * a.height + b.width * b.height - inter;
            if (union <= 0) return 0;
            return inter / union;
        }

        function existingAnnotationRects() {
            const rects = [];
            const currentBboxes = bboxes[currentImage.name];
            if (!currentBboxes) return rects;
            for (let className in currentBboxes) {
                currentBboxes[className].forEach((bbox) => {
                    if (!bbox) return;
                    if (Array.isArray(bbox.points) && bbox.points.length >= 3) {
                        const xs = bbox.points.map((p) => p.x);
                        const ys = bbox.points.map((p) => p.y);
                        const minX = Math.min(...xs);
                        const maxX = Math.max(...xs);
                        const minY = Math.min(...ys);
                        const maxY = Math.max(...ys);
                        rects.push({ x: minX, y: minY, width: maxX - minX, height: maxY - minY });
                    } else {
                        rects.push({ x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height });
                    }
                });
            }
            return rects;
        }

        sam3SimilarityRequestActive = true;
        updateSam3TextButtons();
        setSam3TextStatus("Running SAM3 similarity prompt…", "info");
        setSamStatus("Running SAM3 similarity prompt…", { variant: "info", duration: 0 });
        try {
            const result = await invokeSam3VisualPrompt({
                bbox_left: left,
                bbox_top: top,
                bbox_width: width,
                bbox_height: height,
                threshold,
                mask_threshold: maskThreshold,
                min_size: minSize,
                simplify_epsilon: simplifyEps,
                max_results: maxResults,
            });
            if (currentImage && result?.image_token) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            let detections = Array.isArray(result?.detections) ? result.detections.slice() : [];
            if (detections.length) {
                const exemplarRect = { x: left, y: top, width, height };
                const exemplarCx = exemplarRect.x + exemplarRect.width / 2;
                const exemplarCy = exemplarRect.y + exemplarRect.height / 2;
                const exemplarMaxDim = Math.max(exemplarRect.width, exemplarRect.height);
                const existingRects = existingAnnotationRects();
                detections = detections.filter((det) => {
                    if (!det || !det.bbox) return false;
                    const rect = yoloBoxToPixelRect(det.bbox);
                    if (!rect) return true;
                    const rectCx = rect.x + rect.width / 2;
                    const rectCy = rect.y + rect.height / 2;
                    const dist = Math.hypot(rectCx - exemplarCx, rectCy - exemplarCy);
                    const iou = rectIoU(rect, exemplarRect);
                    const nearCenter = dist <= exemplarMaxDim * 0.1;
                    const highOverlap = iou >= 0.85;
                    if (nearCenter || highOverlap) {
                        return false;
                    }
                    for (const existing of existingRects) {
                        const overlap = rectIoU(rect, existing);
                        if (overlap >= 0.75) {
                            return false;
                        }
                        const exCx = existing.x + existing.width / 2;
                        const exCy = existing.y + existing.height / 2;
                        const exMax = Math.max(existing.width, existing.height);
                        const distEx = Math.hypot(rectCx - exCx, rectCy - exCy);
                        if (distEx <= exMax * 0.1) {
                            return false;
                        }
                    }
                    return true;
                });
            }
            const added = applySegAwareDetections(detections, targetClass, "SAM3 similarity");
            if (added) {
                const shapeLabel = datasetType === "seg" ? "polygon" : "bbox";
                setSam3TextStatus(`Similarity added ${added} ${shapeLabel}${added === 1 ? "" : "es"}.`, "success");
            } else {
                const warning = Array.isArray(result?.warnings) && result.warnings.includes("no_results")
                    ? "SAM3 found no similar objects."
                    : "SAM3 returned no usable detections.";
                setSam3TextStatus(warning, "warn");
            }
        } catch (error) {
            const detail = error?.message || error;
            setSam3TextStatus(`SAM3 similarity error: ${detail}`, "error");
            console.error("SAM3 similarity prompt failed", error);
        } finally {
            sam3SimilarityRequestActive = false;
            updateSam3TextButtons();
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

    async function invokeSam3TextPrompt(requestFields, { auto = false } = {}) {
        if (!currentImage) {
            throw new Error("No active image");
        }
        const imageNameForRequest = currentImage.name;
        const variantForRequest = "sam3";
        const preloadToken = await waitForSamPreloadIfActive(imageNameForRequest, variantForRequest);
        let payload = await buildSamImagePayload({ variantOverride: variantForRequest, preferredToken: preloadToken });
        if (imageNameForRequest && !payload.image_name) {
            payload.image_name = imageNameForRequest;
        }
        payload.sam_variant = variantForRequest;
        let resp = await fetch(`${API_ROOT}${auto ? "/sam3/text_prompt_auto" : "/sam3/text_prompt"}`, {
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
            resp = await fetch(`${API_ROOT}${auto ? "/sam3/text_prompt_auto" : "/sam3/text_prompt"}`, {
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

    async function invokeSam3VisualPrompt(requestFields) {
        if (!currentImage) {
            throw new Error("No active image");
        }
        const imageNameForRequest = currentImage.name;
        const variantForRequest = "sam3";
        const preloadToken = await waitForSamPreloadIfActive(imageNameForRequest, variantForRequest);
        let payload = await buildSamImagePayload({ variantOverride: variantForRequest, preferredToken: preloadToken });
        if (imageNameForRequest && !payload.image_name) {
            payload.image_name = imageNameForRequest;
        }
        payload.sam_variant = variantForRequest;
        let resp = await fetch(`${API_ROOT}/sam3/visual_prompt`, {
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
            resp = await fetch(`${API_ROOT}/sam3/visual_prompt`, {
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
            type: "bbox",
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

    function addPolygonFromYoloRect(yoloBox, className) {
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
        const pts = [
            { x: absX, y: absY },
            { x: absX + absW, y: absY },
            { x: absX + absW, y: absY + absH },
            { x: absX, y: absY + absH },
        ].map(clampPointToImage);
        const xs = pts.map((p) => p.x);
        const ys = pts.map((p) => p.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const bboxRecord = {
            type: "polygon",
            points: pts,
            x: minX,
            y: minY,
            width: Math.max(0, maxX - minX),
            height: Math.max(0, maxY - minY),
            marked: false,
            class: className,
            uuid: generateUUID(),
        };
        stampBboxCreation(bboxRecord);
        if (!bboxes[currentImage.name]) {
            bboxes[currentImage.name] = {};
        }
        if (!bboxes[currentImage.name][className]) {
            bboxes[currentImage.name][className] = [];
        }
        bboxes[currentImage.name][className].push(bboxRecord);
        setDatasetType("seg");
        return bboxRecord;
    }

    function decodePackedMask(maskPayload) {
        if (!maskPayload || typeof maskPayload.counts !== "string") {
            return null;
        }
        const size = Array.isArray(maskPayload.size) ? maskPayload.size : [];
        if (size.length !== 2) {
            return null;
        }
        const height = parseInt(size[0], 10);
        const width = parseInt(size[1], 10);
        if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
            return null;
        }
        let packed;
        try {
            const raw = atob(maskPayload.counts);
            packed = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) {
                packed[i] = raw.charCodeAt(i);
            }
        } catch (error) {
            console.warn("Failed to decode mask payload", error);
            return null;
        }
        const total = width * height;
        const data = new Uint8Array(total);
        let cursor = 0;
        for (let byteIdx = 0; byteIdx < packed.length && cursor < total; byteIdx++) {
            const byte = packed[byteIdx];
            for (let bit = 7; bit >= 0 && cursor < total; bit--) {
                data[cursor++] = (byte >> bit) & 1;
            }
        }
        return { data, width, height };
    }

    function getMinMaskArea() {
        const raw = sam3TextElements.minSizeInput?.value || "0";
        const parsed = parseFloat(raw);
        if (!Number.isFinite(parsed) || parsed < 0) {
            return 0;
        }
        return parsed;
    }

    function getMaxPolygonPoints() {
        const raw = sam3TextElements.maxPointsInput?.value || "500";
        const parsed = parseInt(raw, 10);
        if (!Number.isFinite(parsed) || parsed <= 3) {
            return 500;
        }
        return Math.min(parsed, 5000);
    }

    function getSimplifyEpsilon() {
        const rawSlider = polygonSimplifyInput ? parseFloat(polygonSimplifyInput.value) : null;
        const sliderValid = Number.isFinite(rawSlider);
        const sliderMin = 0;
        const sliderMax = 40;
        const clampedSlider = sliderValid ? Math.max(sliderMin, Math.min(sliderMax, rawSlider)) : null;
        // Invert: slider left (low) => higher detail (lower epsilon), slider right (high) => more simplification.
        const sliderEps = clampedSlider !== null ? sliderMax - clampedSlider : null;
        const rawFallback = sam3TextElements.epsilonInput?.value || "1.0";
        const parsedFallback = parseFloat(rawFallback);
        const fallbackEps = Number.isFinite(parsedFallback) && parsedFallback >= 0 ? parsedFallback : 1.0;
        return sliderEps !== null ? sliderEps : fallbackEps;
    }

    function simplifyPolygonPoints(points, { maxPoints = 400 } = {}) {
        if (!Array.isArray(points) || points.length === 0) {
            return [];
        }
        const deduped = [];
        points.forEach((pt) => {
            const last = deduped[deduped.length - 1];
            if (!last || Math.abs(last.x - pt.x) > 1e-6 || Math.abs(last.y - pt.y) > 1e-6) {
                deduped.push(pt);
            }
        });
        if (deduped.length > 1) {
            const first = deduped[0];
            const last = deduped[deduped.length - 1];
            if (Math.abs(first.x - last.x) < 1e-6 && Math.abs(first.y - last.y) < 1e-6) {
                deduped.pop();
            }
        }
        const reduced = [];
        for (let i = 0; i < deduped.length; i++) {
            const prev = deduped[(i - 1 + deduped.length) % deduped.length];
            const curr = deduped[i];
            const next = deduped[(i + 1) % deduped.length];
            const cross = (curr.x - prev.x) * (next.y - curr.y) - (curr.y - prev.y) * (next.x - curr.x);
            if (Math.abs(cross) > 1e-6) {
                reduced.push(curr);
            }
        }
        let result = reduced;
        if (result.length > maxPoints) {
            const step = Math.ceil(result.length / maxPoints);
            result = result.filter((_, idx) => idx % step === 0);
        }
        return result;
    }

    function polygonArea(points) {
        if (!Array.isArray(points) || points.length < 3) {
            return 0;
        }
        let area = 0;
        for (let i = 0; i < points.length; i++) {
            const j = (i + 1) % points.length;
            area += points[i].x * points[j].y - points[j].x * points[i].y;
        }
        return Math.abs(area / 2);
    }

    function douglasPeucker(points, epsilon) {
        if (!Array.isArray(points) || points.length < 3) {
            return points || [];
        }
        const sqEps = epsilon * epsilon;
        const distSqToSegment = (p, a, b) => {
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            if (dx === 0 && dy === 0) {
                const ddx = p.x - a.x;
                const ddy = p.y - a.y;
                return ddx * ddx + ddy * ddy;
            }
            const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy)));
            const projX = a.x + t * dx;
            const projY = a.y + t * dy;
            const ddx = p.x - projX;
            const ddy = p.y - projY;
            return ddx * ddx + ddy * ddy;
        };
        const simplifySection = (pts, start, end, out) => {
            if (end <= start + 1) {
                return;
            }
            const a = pts[start];
            const b = pts[end];
            let maxDist = -1;
            let idx = -1;
            for (let i = start + 1; i < end; i++) {
                const d = distSqToSegment(pts[i], a, b);
                if (d > maxDist) {
                    maxDist = d;
                    idx = i;
                }
            }
            if (maxDist > sqEps) {
                simplifySection(pts, start, idx, out);
                out.push(pts[idx]);
                simplifySection(pts, idx, end, out);
            }
        };
        const output = [points[0]];
        simplifySection(points, 0, points.length - 1, output);
        output.push(points[points.length - 1]);
        return output;
    }

    function maskPayloadToPolygons(maskPayload, { maxPointsPerPolygon = 500, maxDim = 512, simplifyEpsilon = 1.0 } = {}) {
        const decoded = decodePackedMask(maskPayload);
        if (!decoded) {
            return [];
        }
        let { data, width, height } = decoded;
        let scaleX = 1;
        let scaleY = 1;
        const maxSide = Math.max(width, height);
        if (maxSide > maxDim) {
            const scale = maxSide / maxDim;
            const newW = Math.max(1, Math.round(width / scale));
            const newH = Math.max(1, Math.round(height / scale));
            const resized = new Uint8Array(newW * newH);
            const stride = width;
            for (let y = 0; y < newH; y++) {
                const srcY = Math.min(height - 1, Math.round(y * scale));
                for (let x = 0; x < newW; x++) {
                    const srcX = Math.min(width - 1, Math.round(x * scale));
                    resized[y * newW + x] = data[srcY * stride + srcX] ? 1 : 0;
                }
            }
            scaleX = width / newW;
            scaleY = height / newH;
            data = resized;
            width = newW;
            height = newH;
        }
        const closed = new Uint8Array(width * height);
        const getVal = (x, y) => (y >= 0 && y < height && x >= 0 && x < width && data[y * width + x]) ? 1 : 0;
        // Morphological close (3x3) with fallback to original if it erases everything
        let closedOnes = 0;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let ones = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (getVal(x + dx, y + dy)) ones++;
                    }
                }
                const val = ones >= 5 ? 1 : 0;
                closed[y * width + x] = val;
                closedOnes += val;
            }
        }
        if (closedOnes === 0) {
            for (let i = 0; i < data.length; i++) {
                closed[i] = data[i];
            }
        }
        const visited = new Uint8Array(width * height);
        const components = [];
        const dirs4 = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
        ];
        const floodFill = (sx, sy) => {
            const stack = [[sx, sy]];
            const pixels = [];
            visited[sy * width + sx] = 1;
            while (stack.length) {
                const [cx, cy] = stack.pop();
                pixels.push([cx, cy]);
                for (const [dx, dy] of dirs4) {
                    const nx = cx + dx;
                    const ny = cy + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                    const idx = ny * width + nx;
                    if (visited[idx] || !closed[idx]) continue;
                    visited[idx] = 1;
                    stack.push([nx, ny]);
                }
            }
            return pixels;
        };
        const addEdge = (edgeMap, start, end) => {
            const key = `${start[0]},${start[1]}`;
            let bucket = edgeMap.get(key);
            if (!bucket) {
                bucket = [];
                edgeMap.set(key, bucket);
            }
            bucket.push({ end, used: false });
        };
        const buildPolygonFromEdges = (edgeMap) => {
            const polygons = [];
            for (const [startKey, edges] of edgeMap.entries()) {
                for (const edge of edges) {
                    if (edge.used) continue;
                    const startParts = startKey.split(",").map((v) => parseInt(v, 10));
                    if (startParts.length !== 2 || startParts.some((v) => Number.isNaN(v))) {
                        continue;
                    }
                    const polygon = [];
                    let current = [startParts[0], startParts[1]];
                    let guard = 0;
                    const guardLimit = edgeMap.size * 8 + 1000;
                    while (guard < guardLimit) {
                        polygon.push({ x: current[0] * scaleX, y: current[1] * scaleY });
                        const key = `${current[0]},${current[1]}`;
                        const bucket = edgeMap.get(key);
                        if (!bucket) break;
                        const nextEdge = bucket.find((e) => !e.used);
                        if (!nextEdge) break;
                        nextEdge.used = true;
                        current = nextEdge.end;
                        if (current[0] === startParts[0] && current[1] === startParts[1]) {
                            polygon.push({ x: current[0] * scaleX, y: current[1] * scaleY });
                            break;
                        }
                        guard++;
                    }
                    if (polygon.length >= 3) {
                        const simplified = douglasPeucker(polygon, simplifyEpsilon);
                        let capped = simplified;
                        if (capped.length > maxPointsPerPolygon) {
                            const step = Math.ceil(capped.length / maxPointsPerPolygon);
                            capped = capped.filter((_, idx) => idx % step === 0);
                        }
                        if (capped.length >= 3) {
                            polygons.push(capped);
                        }
                    }
                }
            }
            return polygons;
        };
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                if (!closed[idx] || visited[idx]) continue;
                const pixels = floodFill(x, y);
                components.push(pixels);
            }
        }
        const allPolys = [];
        for (const pixels of components) {
            const compSet = new Set(pixels.map((p) => `${p[0]},${p[1]}`));
            const edgeMap = new Map();
            for (const [px, py] of pixels) {
                const idx = py * width + px;
                const neighbors = [
                    [px, py - 1],
                    [px + 1, py],
                    [px, py + 1],
                    [px - 1, py],
                ];
                const pts = [
                    [px, py],
                    [px + 1, py],
                    [px + 1, py + 1],
                    [px, py + 1],
                ];
                const inside = compSet.has(`${px},${py}`);
                for (let k = 0; k < neighbors.length; k++) {
                    const neighborKey = `${neighbors[k][0]},${neighbors[k][1]}`;
                    const neighborInside = compSet.has(neighborKey);
                    if (inside && !neighborInside) {
                        const start = pts[k];
                        const end = pts[(k + 1) % pts.length];
                        addEdge(edgeMap, start, end);
                    }
                }
            }
            const polys = buildPolygonFromEdges(edgeMap);
            polys.forEach((p) => allPolys.push(p));
        }
        // Deduplicate near-identical polygons
        const deduped = [];
        const seen = new Set();
        for (const poly of allPolys) {
            if (!poly || poly.length < 3) continue;
            const area = polygonArea(poly);
            const key = `${poly.length}:${poly[0].x.toFixed(1)},${poly[0].y.toFixed(1)}:${area.toFixed(1)}`;
            if (seen.has(key)) continue;
            seen.add(key);
            deduped.push(poly);
        }
        return deduped;
    }

    function addPolygonFromMask(maskPayload, className, { simplifyEpsilon = null, maxPointsPerPolygon = null, minArea = null } = {}) {
        if (!currentImage) {
            return null;
        }
        const uiEpsilon = getSimplifyEpsilon();
        const epsilon = Number.isFinite(uiEpsilon) && uiEpsilon >= 0
            ? uiEpsilon
            : (Number.isFinite(simplifyEpsilon) && simplifyEpsilon >= 0 ? simplifyEpsilon : 1.0);
        const maxPts = Number.isFinite(maxPointsPerPolygon) && maxPointsPerPolygon > 3 ? maxPointsPerPolygon : getMaxPolygonPoints();
        const minMaskArea = Number.isFinite(minArea) && minArea >= 0 ? minArea : Math.max(0, getMinMaskArea());
        const polygons = maskPayloadToPolygons(maskPayload, {
            maxPointsPerPolygon: maxPts,
            simplifyEpsilon: epsilon,
        });
        if (!Array.isArray(polygons) || polygons.length === 0) {
            return null;
        }
        const scored = polygons
            .map((pts) => {
                const clamped = pts.map(clampPointToImage);
                return { pts: clamped, area: polygonArea(clamped) };
            })
            .filter((entry) => entry.area > 0);
        if (scored.length === 0) {
            return null;
        }
        scored.sort((a, b) => b.area - a.area);
        const maxPolygons = 1;
        const minAreaThresh = minMaskArea;
        let firstRecord = null;
        if (!bboxes[currentImage.name]) {
            bboxes[currentImage.name] = {};
        }
        if (!bboxes[currentImage.name][className]) {
            bboxes[currentImage.name][className] = [];
        }
        for (let i = 0; i < scored.length && i < maxPolygons; i++) {
            const chosen = scored[i].pts;
            if (!chosen || chosen.length < 3) {
                continue;
            }
            if (scored[i].area < minAreaThresh) {
                continue;
            }
            const xs = chosen.map((p) => p.x);
            const ys = chosen.map((p) => p.y);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            const bboxRecord = {
                type: "polygon",
                points: chosen,
                x: minX,
                y: minY,
                width: Math.max(0, maxX - minX),
                height: Math.max(0, maxY - minY),
                marked: false,
                class: className,
                uuid: generateUUID(),
            };
            stampBboxCreation(bboxRecord);
            bboxes[currentImage.name][className].push(bboxRecord);
            if (!firstRecord) {
                firstRecord = bboxRecord;
            }
        }
        if (firstRecord) {
            setDatasetType("seg");
        }
        return firstRecord;
    }

    function addDetectionAnnotation(entry, className) {
        if (datasetType === "seg") {
            if (entry?.mask) {
                const epsVal = Number(entry.simplify_epsilon);
                const created = addPolygonFromMask(entry.mask, className, {
                    simplifyEpsilon: Number.isFinite(epsVal) && epsVal >= 0 ? epsVal : null,
                });
                if (created) {
                    return created;
                }
            }
            return addPolygonFromYoloRect(entry.bbox, className);
        }
        return addYoloBoxFromQwen(entry.bbox, className);
    }

    function applyDetectionsToClass(entries, className, sourceLabel = "Detector") {
        return applySegAwareDetections(entries, className, sourceLabel);
    }

    function applySegAwareDetections(entries, className, sourceLabel = "Detector") {
        if (!currentImage || !className || !Array.isArray(entries) || entries.length === 0) {
            return 0;
        }
        let added = 0;
        entries.forEach((entry) => {
            if (!entry || !entry.bbox) {
                return;
            }
            const created = addDetectionAnnotation(entry, className);
            if (!created) {
                return;
            }
            if (typeof entry.score === "number") {
                created.samScore = entry.score;
            }
            added += 1;
        });
        if (added > 0) {
            const shapeLabel = datasetType === "seg" ? "polygon" : "bbox";
            setSamStatus(`${sourceLabel} added ${added} ${shapeLabel}${added === 1 ? "" : "es"} to ${className}`, { variant: "success", duration: 4500 });
        }
        return added;
    }

    function applyQwenBoxes(boxes, className) {
        return applySegAwareDetections(boxes, className, "Qwen");
    }

    function applySam3AutoDetections(entries, fallbackClass = null) {
        if (!currentImage || !Array.isArray(entries) || entries.length === 0) {
            return 0;
        }
        let added = 0;
        entries.forEach((entry) => {
            if (!entry || !entry.bbox) {
                return;
            }
            let targetClass = null;
            if (entry.prediction && typeof classes[entry.prediction] !== "undefined") {
                targetClass = entry.prediction;
            } else {
                targetClass = fallbackClass || getQwenTargetClass();
            }
            if (!targetClass) {
                return;
            }
            const created = addDetectionAnnotation(entry, targetClass);
            if (!created) {
                return;
            }
            if (typeof entry.score === "number") {
                created.samScore = entry.score;
            }
            added += 1;
        });
        if (added > 0) {
            const shapeLabel = datasetType === "seg" ? "polygon" : "bbox";
            setSamStatus(`SAM3 auto added ${added} ${shapeLabel}${added === 1 ? "" : "es"}.`, { variant: "success", duration: 4500 });
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
                await samPointMultiAutoPrompt(job, jobHandle);
            } else {
                await samPointMultiPrompt(job, jobHandle);
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
        refreshPolygonDetailVisibility();
        refreshSam3SimilarityVisibility();
        console.log("SAM mode =>", samMode, "samAutoMode =>", samAutoMode);
    }

    function renderAgentResults(result) {
        if (!agentElements.results) return;
        agentElements.results.innerHTML = "";
        if (!result || !Array.isArray(result.classes)) {
            setAgentResultsMessage("No agent mining results yet.", "warn");
            return;
        }
        const frag = document.createDocumentFragment();
        result.classes.forEach((cls) => {
            const card = document.createElement("div");
            card.className = "training-card";
            const body = document.createElement("div");
            body.className = "training-card__body";
            const recipe = cls.recipe || {};
            const steps = Array.isArray(recipe.steps) ? recipe.steps : [];
            const summary = recipe.summary || {};
            const covPct = Number.isFinite(summary.coverage_rate) ? (summary.coverage_rate * 100).toFixed(1) : "0.0";
            body.innerHTML = `
                <div class="training-history-row">
                    <div class="training-history-title">${escapeHtml(cls.name || cls.id)}</div>
                    <span class="badge">${steps.length} step${steps.length === 1 ? "" : "s"}</span>
                </div>
                <div class="training-help">GT train/val: ${cls.train_gt || 0}/${cls.val_gt || 0}</div>
                <div><strong>Coverage:</strong> ${summary.covered || 0}/${summary.total_gt || 0} (${covPct}%) • FPs: ${summary.fps || 0}</div>
            `;
            if (steps.length) {
                const table = document.createElement("table");
                table.className = "training-table";
                table.innerHTML = `
                    <thead>
                        <tr><th>#</th><th>Type</th><th>Prompt/Exemplar</th><th>Thr</th><th>Gain</th><th>FPs</th><th>Cov%</th></tr>
                    </thead>
                `;
                const tbody = document.createElement("tbody");
                steps.forEach((step, idx) => {
                    const covAfter = Number.isFinite(step.coverage_after) ? (step.coverage_after * 100).toFixed(1) : "";
                    const label =
                        step.type === "visual" && step.exemplar
                            ? `Exemplar img ${step.exemplar.image_id} bbox ${Array.isArray(step.exemplar.bbox) ? step.exemplar.bbox.join(",") : ""}`
                            : step.prompt || "";
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td>${idx + 1}</td>
                        <td>${step.type || "text"}</td>
                        <td>${escapeHtml(label)}</td>
                        <td>${(step.threshold ?? "").toString()}</td>
                        <td>${step.gain ?? ""}</td>
                        <td>${step.fps ?? ""}</td>
                        <td>${covAfter}</td>
                    `;
                    tbody.appendChild(row);
                });
                table.appendChild(tbody);
                body.appendChild(table);
            } else {
                const empty = document.createElement("div");
                empty.className = "training-help";
                empty.textContent = "No steps proposed for this class.";
                body.appendChild(empty);
            }
            const foot = document.createElement("div");
            foot.className = "training-actions";
            const saveBtn = document.createElement("button");
            saveBtn.type = "button";
            saveBtn.className = "training-button secondary";
            saveBtn.textContent = "Save recipe";
            saveBtn.addEventListener("click", async () => {
                const datasetId = agentElements.datasetSelect?.value;
                if (!datasetId) {
                    setAgentStatus("Select a dataset before saving.", "warn");
                    return;
                }
                const label = prompt(`Recipe label for ${cls.name || cls.id}?`, `${cls.name || cls.id} recipe`);
                if (!label) return;
                try {
                    const resp = await fetch(`${API_ROOT}/agent_mining/recipes`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            dataset_id: datasetId,
                            class_id: cls.id,
                            class_name: cls.name,
                            label,
                            recipe,
                        }),
                    });
                    if (!resp.ok) throw new Error(await resp.text());
                    setAgentStatus(`Saved recipe "${label}".`, "success");
                    fetchAgentRecipes().catch((err) => console.error("Agent recipe refresh failed", err));
                } catch (err) {
                    console.error("Save recipe failed", err);
                    setAgentStatus(`Save failed: ${err.message || err}`, "error");
                }
            });
            foot.appendChild(saveBtn);
            body.appendChild(foot);
            card.appendChild(body);
            frag.appendChild(card);
        });
        agentElements.results.appendChild(frag);
    }

    function renderAgentLogs(job) {
        if (!agentElements.logs) return;
        agentElements.logs.innerHTML = "";
        if (!job || !Array.isArray(job.logs)) return;
        const frag = document.createDocumentFragment();
        job.logs.slice(-200).forEach((entry) => {
            const div = document.createElement("div");
            div.className = "training-log-line";
            const ts = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : "";
            div.textContent = `${ts ? `[${ts}] ` : ""}${entry.msg || entry.message || entry}`;
            frag.appendChild(div);
        });
        agentElements.logs.appendChild(frag);
    }

    function updateAgentProgress(job) {
        if (!agentElements.progressFill) return;
        const pct = Number.isFinite(job?.progress) ? Math.max(0, Math.min(1, job.progress)) * 100 : 0;
        agentElements.progressFill.style.width = `${pct}%`;
    }

    function getAgentSelectedDatasetMeta() {
        const datasetId = agentElements.datasetSelect?.value;
        if (!datasetId) return null;
        return agentState.datasetsById[datasetId] || null;
    }

    function renderAgentRecipeDetails(recipeData) {
        if (!agentElements.recipeDetails) return;
        agentElements.recipeDetails.innerHTML = "";
        agentElements.recipeDetails.style.display = recipeData ? "block" : "none";
        if (!recipeData) return;
        const dsMeta = getAgentSelectedDatasetMeta();
        const dsClasses = Array.isArray(dsMeta?.classes) ? dsMeta.classes : [];
        const recipeClasses = Array.isArray(recipeData.labelmap) ? recipeData.labelmap : [];
        const mismatchWarnings = [];
        if (dsMeta?.signature && recipeData.dataset_signature && dsMeta.signature !== recipeData.dataset_signature) {
            mismatchWarnings.push("Dataset signature differs from the recipe origin.");
        }
        if (recipeClasses.length && dsClasses.length && JSON.stringify(recipeClasses) !== JSON.stringify(dsClasses)) {
            mismatchWarnings.push("Label map differs; remap the class before applying.");
        }
        const wrapper = document.createElement("div");
        wrapper.className = "training-card__body";
        const title = document.createElement("div");
        title.className = "training-history-title";
        title.textContent = recipeData.label || recipeData.id || "recipe";
        wrapper.appendChild(title);
            const meta = document.createElement("div");
            meta.className = "training-help";
            meta.textContent = [
                recipeData.class_name ? `Class: ${recipeData.class_name}` : null,
                recipeData.dataset_id ? `Dataset: ${recipeData.dataset_id}` : null,
                recipeData.dataset_signature ? `Signature: ${recipeData.dataset_signature}` : null,
                recipeData.labelmap_hash ? `Labelmap hash: ${recipeData.labelmap_hash}` : null,
            ]
                .filter(Boolean)
                .join(" • ");
            wrapper.appendChild(meta);
        if (mismatchWarnings.length) {
            const warn = document.createElement("div");
            warn.className = "training-message warn";
            warn.textContent = mismatchWarnings.join(" ");
            wrapper.appendChild(warn);
        }
        if (dsClasses.length) {
            const mapRow = document.createElement("div");
            mapRow.className = "training-field";
            const lbl = document.createElement("label");
            lbl.textContent = "Remap class";
            mapRow.appendChild(lbl);
            const select = document.createElement("select");
            dsClasses.forEach((clsName, idx) => {
                const opt = document.createElement("option");
                opt.value = (idx + 1).toString();
                opt.textContent = clsName;
                if (recipeData.class_name && clsName.toLowerCase() === recipeData.class_name.toLowerCase()) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            });
            select.addEventListener("change", () => {
                const val = parseInt(select.value, 10);
                if (Number.isInteger(val)) {
                    agentState.recipeClassOverride = { class_id: val, class_name: dsClasses[val - 1] };
                }
            });
            mapRow.appendChild(select);
            wrapper.appendChild(mapRow);
        }
        const steps = recipeData.recipe?.steps || [];
        if (steps.length) {
            const list = document.createElement("div");
            list.className = "training-grid";
            steps.forEach((step, idx) => {
                const card = document.createElement("div");
                card.className = "training-card";
                const body = document.createElement("div");
                body.className = "training-card__body";
                const header = document.createElement("div");
                header.className = "training-history-row";
                header.innerHTML = `<div class="training-history-title">Step ${idx + 1}: ${escapeHtml(step.type || "text")}</div><span class="badge">thr ${step.threshold ?? ""}</span>`;
                body.appendChild(header);
                const label = document.createElement("div");
                label.className = "training-help";
                label.textContent = step.prompt || (step.exemplar ? "Exemplar" : "");
                body.appendChild(label);
                if (step.exemplar && step.exemplar.crop_base64) {
                    const img = document.createElement("img");
                    img.src = step.exemplar.crop_base64;
                    img.alt = "Exemplar crop";
                    img.style.maxWidth = "120px";
                    img.style.display = "block";
                    img.style.marginTop = "6px";
                    body.appendChild(img);
                }
                card.appendChild(body);
                list.appendChild(card);
            });
            wrapper.appendChild(list);
        }
        agentElements.recipeDetails.appendChild(wrapper);
    }

    async function loadAgentDatasets() {
        if (!agentElements.datasetSelect || !agentElements.datasetSummary) return;
        agentElements.datasetSelect.disabled = true;
        agentElements.datasetSelect.innerHTML = "";
        agentElements.datasetSummary.textContent = "Loading datasets…";
        try {
            const resp = await fetch(`${API_ROOT}/sam3/datasets`);
            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json();
            const list = Array.isArray(data) ? data : [];
            agentState.datasetsById = {};
            agentElements.datasetSelect.innerHTML = "";
            list.forEach((entry) => {
                const opt = document.createElement("option");
                opt.value = entry.id || entry.label || entry.path || "";
                opt.textContent = entry.label || entry.id || opt.value;
                agentElements.datasetSelect.appendChild(opt);
                if (opt.value) {
                    agentState.datasetsById[opt.value] = entry;
                }
            });
            if (list.length) {
                agentElements.datasetSelect.selectedIndex = 0;
                const first = list[0];
                const parts = [];
                if (first.source) parts.push(first.source);
                if (first.image_count) parts.push(`${first.image_count} images`);
                agentElements.datasetSummary.textContent = parts.join(" • ") || "Dataset ready";
            } else {
                agentElements.datasetSummary.textContent = "No datasets found.";
            }
        } catch (err) {
            console.error("Agent datasets load failed", err);
            agentElements.datasetSummary.textContent = `Failed: ${err.message || err}`;
        } finally {
            agentElements.datasetSelect.disabled = false;
        }
    }

    function updateAgentDatasetSummary() {
        if (!agentElements.datasetSummary) return;
        const meta = getAgentSelectedDatasetMeta();
        if (!meta) {
            agentElements.datasetSummary.textContent = "Pick a converted SAM3/Qwen dataset.";
            return;
        }
        const parts = [];
        if (meta.source) parts.push(meta.source);
        if (meta.image_count) parts.push(`${meta.image_count} images`);
        agentElements.datasetSummary.textContent = parts.join(" • ") || "Dataset ready";
    }

    function parseAgentPayload() {
        const datasetId = agentElements.datasetSelect?.value;
        if (!datasetId) {
            setAgentStatus("Select a dataset.", "warn");
            return null;
        }
        const valPct = readNumberInput(agentElements.valPercent, { integer: false });
        const thresholds = parseCsvNumbers(agentElements.thresholds?.value, { clampMin: 0, clampMax: 1 });
        const maskThreshold = readNumberInput(agentElements.maskThreshold, { integer: false });
        const maxResults = readNumberInput(agentElements.maxResults, { integer: true });
        const minSize = readNumberInput(agentElements.minSize, { integer: true });
        const simplifyEps = readNumberInput(agentElements.simplifyEps, { integer: false });
        const maxWorkers = readNumberInput(agentElements.maxWorkers, { integer: true });
        const workersPerGpu = readNumberInput(agentElements.workersPerGpu, { integer: true });
        const exemplars = readNumberInput(agentElements.exemplars, { integer: true });
        const similarityFloor = readNumberInput(agentElements.similarityScore, { integer: false });
        const classesRaw = agentElements.classesInput?.value || "";
        const classes =
            classesRaw
                .split(/[,\\s]+/)
                .map((p) => parseInt(p.trim(), 10))
                .filter((v) => Number.isInteger(v)) || [];
        const hintsRaw = agentElements.classHints?.value || "";
        let classHints = null;
        if (hintsRaw.trim()) {
            try {
                const parsed = JSON.parse(hintsRaw);
                if (parsed && typeof parsed === "object") {
                    classHints = parsed;
                }
            } catch (err) {
                console.warn("Invalid class hints JSON", err);
            }
        }
return {
            dataset_id: datasetId,
            val_percent: Number.isFinite(valPct) ? Math.max(5, Math.min(95, valPct)) / 100 : 0.3,
            thresholds: thresholds.length ? thresholds : [0.2],
            mask_threshold: Number.isFinite(maskThreshold) ? Math.max(0, Math.min(1, maskThreshold)) : 0.5,
            max_results: Number.isFinite(maxResults) ? Math.max(1, maxResults) : 100,
            min_size: Number.isFinite(minSize) ? Math.max(0, minSize) : 0,
            simplify_epsilon: Number.isFinite(simplifyEps) ? Math.max(0, simplifyEps) : 0,
            max_workers: Number.isFinite(maxWorkers) ? Math.max(1, Math.min(16, maxWorkers)) : 1,
            max_workers_per_device: Number.isFinite(workersPerGpu) ? Math.max(1, Math.min(8, workersPerGpu)) : 1,
            exemplar_per_class: Number.isFinite(exemplars) ? Math.max(0, exemplars) : 4,
            cluster_exemplars: !!(agentElements.clusterExemplars && agentElements.clusterExemplars.checked),
            use_clip_fp_guard: !!(agentElements.clipGuard && agentElements.clipGuard.checked),
            similarity_score: Number.isFinite(similarityFloor) ? Math.max(0, Math.min(1, similarityFloor)) : 0.25,
            classes: classes.length ? classes : null,
            class_hints: Object.keys(classHints).length ? classHints : null,
            auto_mine_prompts: (agentElements.qwenMaxPrompts && readNumberInput(agentElements.qwenMaxPrompts, { integer: true }) > 0) || false,
            qwen_max_prompts: Number.isFinite(readNumberInput(agentElements.qwenMaxPrompts, { integer: true }))
                ? Math.max(0, readNumberInput(agentElements.qwenMaxPrompts, { integer: true }))
                : 0,
            test_mode: !!(agentElements.testMode && agentElements.testMode.checked),
            test_train_limit: readNumberInput(agentElements.trainLimit, { integer: true }) ?? 10,
            test_val_limit: readNumberInput(agentElements.valLimit, { integer: true }) ?? 10,
        };
    }

    async function startAgentMiningJob() {
        const payload = parseAgentPayload();
        if (!payload) return;
        setAgentStatus("Starting agent mining…", "info");
        if (agentElements.runButton) agentElements.runButton.disabled = true;
        updateAgentProgress({ progress: 0 });
        stopAgentPoll();
        try {
            const resp = await fetch(`${API_ROOT}/agent_mining/jobs`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) throw new Error(await resp.text());
            const job = await resp.json();
            agentState.lastJob = job;
            setAgentStatus(`Job ${job.job_id} started`, "success");
            updateAgentProgress(job);
            scheduleAgentPoll();
        } catch (err) {
            console.error("Agent mining start failed", err);
            setAgentStatus(`Start failed: ${err.message || err}`, "error");
        } finally {
            if (agentElements.runButton) agentElements.runButton.disabled = false;
        }
    }

    function stopAgentPoll() {
        if (agentState.pollTimer) {
            clearInterval(agentState.pollTimer);
            agentState.pollTimer = null;
        }
        agentState.pollInFlight = false;
    }

    function scheduleAgentPoll() {
        stopAgentPoll();
        agentState.pollTimer = setInterval(() => {
            refreshAgentLatest({ silent: true }).catch((err) => console.error("Agent mining refresh failed", err));
        }, 2000);
    }

    async function refreshAgentLatest(options = {}) {
        const { silent = false } = options;
        if (agentState.pollInFlight) return;
        agentState.pollInFlight = true;
        if (!silent) setAgentStatus("Fetching latest result…", "info");
        if (!silent && agentElements.refreshButton) agentElements.refreshButton.disabled = true;
        try {
            const jobId = agentState.lastJob?.job_id;
            const url = jobId ? `${API_ROOT}/agent_mining/jobs/${jobId}` : `${API_ROOT}/agent_mining/results/latest`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(await resp.text());
            const job = await resp.json();
            agentState.lastJob = job;
            if (!silent) setAgentStatus(`Latest job: ${job.status}`, "success");
            updateAgentProgress(job);
            renderAgentResults(job.result);
            renderAgentLogs(job);
            const keepPolling = !["completed", "failed", "cancelled"].includes(job.status || "");
            if (keepPolling) {
                scheduleAgentPoll();
            } else {
                stopAgentPoll();
            }
        } catch (err) {
            console.error("Agent mining latest failed", err);
            setAgentResultsMessage(`Fetch failed: ${err.message || err}`, "error");
            setAgentStatus(`Fetch failed: ${err.message || err}`, "error");
            stopAgentPoll();
        } finally {
            agentState.pollInFlight = false;
            if (!silent && agentElements.refreshButton) agentElements.refreshButton.disabled = false;
        }
    }

    async function cancelAgentJob() {
        const jobId = agentState.lastJob?.job_id;
        if (!jobId) {
            setAgentStatus("No running job to cancel.", "warn");
            return;
        }
        setAgentStatus("Cancelling job…", "info");
        try {
            const resp = await fetch(`${API_ROOT}/agent_mining/jobs/${jobId}/cancel`, { method: "POST" });
            if (!resp.ok) throw new Error(await resp.text());
            const job = await resp.json();
            agentState.lastJob = job;
            setAgentStatus(`Job ${jobId} cancelled`, "success");
            updateAgentProgress(job);
            renderAgentLogs(job);
            stopAgentPoll();
        } catch (err) {
            console.error("Agent cancel failed", err);
            setAgentStatus(`Cancel failed: ${err.message || err}`, "error");
        }
    }

    async function fetchAgentRecipes() {
        const datasetId = agentElements.datasetSelect?.value;
        if (!agentElements.recipeSelect) return;
        agentElements.recipeSelect.innerHTML = "";
        try {
            const url = new URL(`${API_ROOT}/agent_mining/recipes`);
            if (datasetId) url.searchParams.set("dataset_id", datasetId);
            const resp = await fetch(url.toString());
            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json();
            const list = Array.isArray(data) ? data : [];
            list.forEach((rec) => {
                const opt = document.createElement("option");
                opt.value = rec.id || "";
                const labelParts = [];
                if (rec.label) labelParts.push(rec.label);
                if (rec.class_name) labelParts.push(`(${rec.class_name})`);
                opt.textContent = labelParts.join(" ") || rec.id;
                agentElements.recipeSelect.appendChild(opt);
            });
            if (!list.length) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "No saved recipes";
                agentElements.recipeSelect.appendChild(opt);
            }
        } catch (err) {
            console.error("Fetch recipes failed", err);
            setAgentStatus(`Recipe list failed: ${err.message || err}`, "error");
        }
    }

    async function loadSelectedAgentRecipe() {
        const recipeId = agentElements.recipeSelect?.value;
        if (!recipeId) {
            setAgentStatus("Select a recipe to load.", "warn");
            return;
        }
        try {
            const resp = await fetch(`${API_ROOT}/agent_mining/recipes/${recipeId}`);
            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json();
            const recipe = data.recipe || {};
            agentState.lastJob = {
                status: "completed",
                message: `Loaded recipe ${data.label || data.id}`,
                result: {
                    classes: [
                        {
                            id: data.class_id,
                            name: data.class_name,
                            train_gt: null,
                            val_gt: null,
                            exemplars: [],
                            clip_warnings: [],
                            candidates: [],
                            recipe,
                            coverage_by_image: null,
                        },
                    ],
                },
            };
            agentState.loadedRecipe = data;
            agentState.recipeClassOverride = null;
            renderAgentResults(agentState.lastJob.result);
            renderAgentRecipeDetails(data);
            renderAgentLogs(null);
            setAgentStatus(`Loaded recipe ${data.label || data.id}`, "success");
        } catch (err) {
            console.error("Load recipe failed", err);
            setAgentStatus(`Load failed: ${err.message || err}`, "error");
        }
    }

    async function downloadSelectedAgentRecipe() {
        const recipeId = agentElements.recipeSelect?.value;
        if (!recipeId) {
            setAgentStatus("Select a recipe to download.", "warn");
            return;
        }
        try {
            const resp = await fetch(`${API_ROOT}/agent_mining/recipes/${recipeId}/export`);
            if (!resp.ok) throw new Error(await resp.text());
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${recipeId}.zip`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            setAgentStatus("Downloaded recipe package.", "success");
        } catch (err) {
            console.error("Recipe download failed", err);
            setAgentStatus(`Download failed: ${err.message || err}`, "error");
        }
    }

    async function importAgentRecipe() {
        if (!agentElements.recipeFile || !agentElements.recipeFile.files?.length) {
            setAgentStatus("Choose a recipe file (.zip or .json) to import.", "warn");
            return;
        }
        const file = agentElements.recipeFile.files[0];
        const formData = new FormData();
        formData.append("file", file);
        setAgentStatus("Importing recipe…", "info");
        try {
            const resp = await fetch(`${API_ROOT}/agent_mining/recipes/import`, {
                method: "POST",
                body: formData,
            });
            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json();
            agentState.loadedRecipe = data;
            agentState.recipeClassOverride = null;
            renderAgentRecipeDetails(data);
            setAgentStatus(`Imported recipe ${data.label || data.id}`, "success");
            await fetchAgentRecipes();
        } catch (err) {
            console.error("Recipe import failed", err);
            setAgentStatus(`Import failed: ${err.message || err}`, "error");
        } finally {
            agentElements.recipeFile.value = "";
        }
    }

    async function applySelectedAgentRecipe() {
        const recipeId = agentElements.recipeSelect?.value;
        const datasetId = agentElements.datasetSelect?.value;
        if (!datasetId) {
            setAgentStatus("Select a dataset first.", "warn");
            return;
        }
        if (!recipeId) {
            setAgentStatus("Select a recipe to apply.", "warn");
            return;
        }
        const imageId = readNumberInput(agentElements.recipeImageId, { integer: true });
        if (!Number.isInteger(imageId) || imageId < 0) {
            setAgentStatus("Enter a valid COCO image id to apply.", "warn");
            return;
        }
        try {
            let recipeData = agentState.loadedRecipe;
            if (!recipeData || recipeData.id !== recipeId) {
                const recipeResp = await fetch(`${API_ROOT}/agent_mining/recipes/${recipeId}`);
                if (!recipeResp.ok) throw new Error(await recipeResp.text());
                recipeData = await recipeResp.json();
            }
            if (agentState.recipeClassOverride) {
                recipeData = {
                    ...recipeData,
                    class_id: agentState.recipeClassOverride.class_id,
                    class_name: agentState.recipeClassOverride.class_name,
                };
            }
            const payload = {
                dataset_id: datasetId,
                image_id: imageId,
                recipe: recipeData,
                mask_threshold: recipeData.params?.mask_threshold ?? 0.5,
                min_size: recipeData.params?.min_size ?? 0,
                simplify_epsilon: recipeData.params?.simplify_epsilon ?? 0.5,
                max_results: recipeData.params?.max_results ?? 100,
            };
            const resp = await fetch(`${API_ROOT}/agent_mining/apply`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json();
            const detections = Array.isArray(data?.detections) ? data.detections : [];
            const warnList = Array.isArray(data?.warnings) ? data.warnings : [];
            const warnText = warnList.length ? ` • Warnings: ${warnList.join(", ")}` : "";
            setAgentStatus(`Apply succeeded: ${detections.length} detections${warnText}`, warnList.length ? "warn" : "success");
            // Optionally overlay on current view? Not wired to the annotator yet.
        } catch (err) {
            console.error("Agent recipe apply failed", err);
            setAgentStatus(`Apply failed: ${err.message || err}`, "error");
        }
    }

    function initAgentMiningUi() {
        agentElements.datasetSelect = document.getElementById("agentDatasetSelect");
        agentElements.datasetRefresh = document.getElementById("agentDatasetRefresh");
        agentElements.datasetSummary = document.getElementById("agentDatasetSummary");
        agentElements.valPercent = document.getElementById("agentValPercent");
        agentElements.thresholds = document.getElementById("agentThresholds");
        agentElements.maskThreshold = document.getElementById("agentMaskThreshold");
        agentElements.maxResults = document.getElementById("agentMaxResults");
        agentElements.minSize = document.getElementById("agentMinSize");
        agentElements.simplifyEps = document.getElementById("agentSimplifyEps");
        agentElements.maxWorkers = document.getElementById("agentMaxWorkers");
        agentElements.workersPerGpu = document.getElementById("agentWorkersPerGpu");
        agentElements.exemplars = document.getElementById("agentExemplars");
        agentElements.clusterExemplars = document.getElementById("agentClusterExemplars");
        agentElements.clipGuard = document.getElementById("agentClipGuard");
        agentElements.similarityScore = document.getElementById("agentSimilarityScore");
        agentElements.classesInput = document.getElementById("agentClasses");
        agentElements.classHints = document.getElementById("agentClassHints");
        agentElements.qwenMaxPrompts = document.getElementById("agentQwenMaxPrompts");
        agentElements.testMode = document.getElementById("agentTestMode");
        agentElements.trainLimit = document.getElementById("agentTrainLimit");
        agentElements.valLimit = document.getElementById("agentValLimit");
        agentElements.runButton = document.getElementById("agentRunBtn");
        agentElements.refreshButton = document.getElementById("agentRefreshBtn");
        agentElements.cancelButton = document.getElementById("agentCancelBtn");
        agentElements.status = document.getElementById("agentStatus");
        agentElements.results = document.getElementById("agentResults");
        agentElements.recipeSelect = document.getElementById("agentRecipeSelect");
        agentElements.recipeRefresh = document.getElementById("agentRecipeRefresh");
        agentElements.recipeLoad = document.getElementById("agentRecipeLoad");
        agentElements.recipeDownload = document.getElementById("agentRecipeDownload");
        agentElements.recipeApply = document.getElementById("agentRecipeApply");
        agentElements.recipeImageId = document.getElementById("agentRecipeImageId");
        agentElements.recipeImport = document.getElementById("agentRecipeImport");
        agentElements.recipeFile = document.getElementById("agentRecipeFile");
        agentElements.recipeDetails = document.getElementById("agentRecipeDetails");
        agentElements.logs = document.getElementById("agentLogs");
        agentElements.progressFill = document.getElementById("agentProgressFill");
        if (agentElements.logs) agentElements.logs.innerHTML = "";
        if (agentElements.progressFill) agentElements.progressFill.style.width = "0%";
        if (agentElements.clusterExemplars) agentElements.clusterExemplars.checked = true;
        if (agentElements.clipGuard) agentElements.clipGuard.checked = true;
        stopAgentPoll();
        if (agentElements.datasetRefresh) {
            agentElements.datasetRefresh.addEventListener("click", () => loadAgentDatasets());
        }
        if (agentElements.datasetSelect) {
            agentElements.datasetSelect.addEventListener("change", () => {
                updateAgentDatasetSummary();
                agentState.loadedRecipe = null;
                agentState.recipeClassOverride = null;
                renderAgentRecipeDetails(null);
                fetchAgentRecipes().catch((err) => console.error("Agent recipe refresh failed", err));
            });
        }
        if (agentElements.runButton) {
            agentElements.runButton.addEventListener("click", () => startAgentMiningJob().catch((err) => console.error("Agent mining start failed", err)));
        }
        if (agentElements.refreshButton) {
            agentElements.refreshButton.addEventListener("click", () => refreshAgentLatest().catch((err) => console.error("Agent mining refresh failed", err)));
        }
        if (agentElements.cancelButton) {
            agentElements.cancelButton.addEventListener("click", () => cancelAgentJob().catch((err) => console.error("Agent mining cancel failed", err)));
        }
        if (agentElements.recipeRefresh) {
            agentElements.recipeRefresh.addEventListener("click", () => fetchAgentRecipes().catch((err) => console.error("Agent recipe refresh failed", err)));
        }
        if (agentElements.recipeLoad) {
            agentElements.recipeLoad.addEventListener("click", () => loadSelectedAgentRecipe().catch((err) => console.error("Agent recipe load failed", err)));
        }
        if (agentElements.recipeDownload) {
            agentElements.recipeDownload.addEventListener("click", () => downloadSelectedAgentRecipe().catch((err) => console.error("Agent recipe download failed", err)));
        }
        if (agentElements.recipeApply) {
            agentElements.recipeApply.addEventListener("click", () => applySelectedAgentRecipe().catch((err) => console.error("Agent recipe apply failed", err)));
        }
        if (agentElements.recipeImport && agentElements.recipeFile) {
            agentElements.recipeImport.addEventListener("click", () => importAgentRecipe().catch((err) => console.error("Agent recipe import failed", err)));
        }
        loadAgentDatasets().catch((err) => console.error("Agent dataset load failed", err));
        fetchAgentRecipes().catch((err) => console.error("Agent recipe init failed", err));
    }

    document.addEventListener("DOMContentLoaded", () => {
        autoModeCheckbox = document.getElementById("autoMode");
        samModeCheckbox = document.getElementById("samMode");
        pointModeCheckbox = document.getElementById("pointMode");
        multiPointModeCheckbox = document.getElementById("multiPointMode");
        samVariantSelect = document.getElementById("samVariant");
        samPreloadCheckbox = document.getElementById("samPreload");
        polygonSimplifyInput = document.getElementById("polygonSimplifyEpsilon");
        polygonSimplifyField = document.getElementById("polygonSimplifyField");
        imagesSelectButton = document.getElementById("imagesSelect");
        classesSelectButton = document.getElementById("classesSelect");
        bboxesSelectButton = document.getElementById("bboxesSelect");
        bboxesFolderSelectButton = document.getElementById("bboxesSelectFolder");
        samStatusEl = document.getElementById("samStatus");
        samStatusProgressEl = document.getElementById("samStatusProgress");
        polygonDrawToggle = document.getElementById("polygonDrawToggle");
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
        initAgentMiningUi();

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

        if (polygonDrawToggle) {
            polygonDrawToggle.addEventListener("click", () => {
                if (datasetType !== "seg") {
                    setDatasetType("seg");
                    setPolygonDrawEnabled(true, { silent: true });
                } else {
                    setPolygonDrawEnabled(!polygonDrawEnabled);
                }
            });
        }
        if (polygonSimplifyInput) {
            polygonSimplifyInput.addEventListener("input", () => {
                const val = parseFloat(polygonSimplifyInput.value);
                const msg = Number.isFinite(val)
                    ? `Polygon detail: ${val.toFixed(1)} (left = simpler, right = more detail)`
                    : "Polygon detail";
                setSamStatus(msg, { variant: "info", duration: 1500 });
            });
        }
        refreshPolygonDetailVisibility();

        if (samPreloadCheckbox) {
            samPreloadCheckbox.addEventListener("change", () => {
                updateSamPreloadState(samPreloadCheckbox.checked);
            });
        }


        if (samVariantSelect) {
            samVariant = samVariantSelect.value || "sam1";
            updateSam3TextButtons();
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
                updateSam3TextButtons();
                refreshSam3SimilarityVisibility();
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
        setPolygonDrawEnabled(datasetType === "seg", { silent: true });
        applyDatasetModeConstraints();
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

    function applySamResultToSegDataset(result, targetBbox, classNameOverride = null) {
        if (!currentImage) return false;
        const className = classNameOverride || (targetBbox ? targetBbox.class : currentClass);
        if (!className) return false;
        const epsVal = Number.isFinite(Number(result?.simplify_epsilon)) && Number(result.simplify_epsilon) >= 0
            ? Number(result.simplify_epsilon)
            : null;
        let created = null;
        if (result?.mask) {
            created = addPolygonFromMask(result.mask, className, { simplifyEpsilon: epsVal });
        }
        if (!created && result?.bbox) {
            created = addPolygonFromYoloRect(result.bbox, className);
        }
        if (!created) {
            return false;
        }
        const bucket = bboxes[currentImage.name]?.[className] || [];
        const idx = bucket.indexOf(created);
        currentBbox = {
            bbox: created,
            index: idx >= 0 ? idx : bucket.length - 1,
            originalX: created.x,
            originalY: created.y,
            originalWidth: created.width,
            originalHeight: created.height,
            moving: false,
            resizing: null,
        };
        return true;
    }

    /*****************************************************
     * Existing SAM / CLIP calls
     *****************************************************/
    async function samBboxPrompt(bbox) {
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
            let resp = await postSamEndpoint(`${API_ROOT}/sam_bbox`, bodyFields);
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
            if (datasetType === "seg") {
                const applied = applySamResultToSegDataset(result, targetBbox, targetBbox?.class);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                delete pendingApiBboxes[returnedUUID];
                if (!applied) {
                    setSamStatus("SAM returned no mask/bbox to apply.", { variant: "warn", duration: 3000 });
                }
                return;
            }
            if (datasetType === "seg") {
                const applied = applySamResultToSegDataset(result, targetBbox, targetBbox?.class);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                delete pendingApiBboxes[returnedUUID];
                if (!applied) {
                    setSamStatus("SAM returned no mask/bbox to apply.", { variant: "warn", duration: 3000 });
                }
                return;
            }
            if (datasetType === "seg") {
                const applied = applySamResultToSegDataset(result, targetBbox, targetBbox?.class);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                delete pendingApiBboxes[returnedUUID];
                if (!applied) {
                    setSamStatus("SAM auto point returned no mask/bbox to apply.", { variant: "warn", duration: 3000 });
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
                console.warn("No 'bbox' field returned from sam_bbox. Full response:", result);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                return;
            }
            delete pendingApiBboxes[returnedUUID];
        } catch (err) {
            console.error("sam_bbox error:", err);
            alert("sam_bbox call failed: " + err);
            if (placeholderContext) {
                removePendingBbox(placeholderContext);
            }
        } finally {
            completeSamJob(jobHandle.id);
            endSamActionStatus(statusToken);
        }
    }

    async function samPointPrompt(px, py) {
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
            let resp = await postSamEndpoint(`${API_ROOT}/sam_point`, bodyFields);
            if (!resp.ok) {
                throw new Error("sam_point failed: " + resp.statusText);
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
            if (datasetType === "seg") {
                const applied = applySamResultToSegDataset(result, targetBbox, targetBbox?.class);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                delete pendingApiBboxes[returnedUUID];
                if (!applied) {
                    setSamStatus("SAM auto bbox returned no mask/bbox to apply.", { variant: "warn", duration: 3000 });
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
                console.warn("No 'bbox' field in sam_point response:", result);
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
            console.error("samPointPrompt error:", err);
            alert("samPointPrompt call failed: " + err);
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
        const resp = await fetch(`${API_ROOT}/predict_base64`, {
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

    async function samBboxAutoPrompt(bbox) {
        if (datasetType === "seg") {
            setSamStatus("Auto bbox prompt is disabled in polygon mode.", { variant: "warn", duration: 4000 });
            return;
        }
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
            let resp = await postSamEndpoint(`${API_ROOT}/sam_bbox_auto`, bodyData);
            if (!resp.ok) {
                throw new Error("sam_bbox_auto failed: " + resp.statusText);
            }
            const result = await resp.json();
            console.log("sam_bbox_auto =>", result);
            if (!isSamJobActive(jobHandle)) {
                return;
            }
            if (result.image_token && currentImage) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            if (!result.uuid || !result.bbox || result.bbox.length < 4) {
                alert("Auto mode error: missing 'uuid' or invalid 'bbox' in /sam_bbox_auto response.");
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
            console.error("sam_bbox_auto error:", err);
            alert("sam_bbox_auto call failed: " + err);
            if (placeholderContext) {
                removePendingBbox(placeholderContext);
            }
        } finally {
            completeSamJob(jobHandle.id);
            endSamActionStatus(statusToken);
        }
    }

    async function samPointAutoPrompt(px, py) {
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
            let resp = await postSamEndpoint(`${API_ROOT}/sam_point_auto`, bodyData);
            if (!resp.ok) {
                throw new Error("sam_point_auto failed: " + resp.statusText);
            }
            const result = await resp.json();
            console.log("sam_point_auto =>", result);
            if (!isSamJobActive(jobHandle)) {
                return;
            }
            if (result.image_token && currentImage) {
                rememberSamToken(currentImage.name, samVariant, result.image_token);
            }
            if (!result.uuid || !result.bbox || result.bbox.length < 4) {
                alert("Auto mode error: missing 'uuid' or invalid 'bbox' in /sam_point_auto response.");
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
            console.error("sam_point_auto error:", err);
            alert("sam_point_auto call failed: " + err);
            if (placeholderContext) {
                removePendingBbox(placeholderContext);
            }
        } finally {
            completeSamJob(jobHandle.id);
            endSamActionStatus(statusToken);
        }
    }

    async function samPointMultiPrompt(job, jobHandle) {
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
            const resp = await postSamEndpoint(`${API_ROOT}/sam_point_multi`, bodyData);
            if (!resp.ok) {
                throw new Error("sam_point_multi failed: " + resp.statusText);
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
            if (datasetType === "seg") {
                const applied = applySamResultToSegDataset(result, targetBbox, targetBbox?.class);
                if (placeholderContext) {
                    removePendingBbox(placeholderContext);
                }
                delete pendingApiBboxes[returnedUUID];
                if (!applied) {
                    setSamStatus("SAM multi-point returned no mask/bbox to apply.", { variant: "warn", duration: 3000 });
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
                resizing: null,
            };
            if (!result.bbox || result.bbox.length < 4) {
                console.warn("No 'bbox' field in sam_point_multi response:", result);
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
            console.error("sam_point_multi error:", err);
            alert("sam_point_multi call failed: " + err);
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

    async function samPointMultiAutoPrompt(job, jobHandle) {
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
            const resp = await postSamEndpoint(`${API_ROOT}/sam_point_multi_auto`, bodyData);
            if (!resp.ok) {
                throw new Error("sam_point_multi_auto failed: " + resp.statusText);
            }
            const result = await resp.json();
            console.log("sam_point_multi_auto =>", result);
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
                alert("Auto mode error: missing 'uuid' or invalid 'bbox' in /sam_point_multi_auto response.");
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
            console.error("sam_point_multi_auto error:", err);
            alert("sam_point_multi_auto call failed: " + err);
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
    let datasetType = "bbox"; // "bbox" or "seg"
    let datasetTypeBadge = null;
    let polygonSimplifyInput = null;
    let polygonSimplifyField = null;
    let bboxCreationCounter = 0;
    let polygonDraft = null; // {points: [{x,y}], className}
    let polygonDrag = null; // {bbox, className, index, vertexIndex}
    let polygonDrawEnabled = true;
    let polygonDrawToggle = null;

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

    const setPolygonDrawEnabled = (nextEnabled, { silent = false } = {}) => {
        const normalized = Boolean(nextEnabled) && datasetType === "seg";
        polygonDrawEnabled = normalized;
        if (polygonDrawToggle) {
            const label = normalized ? "Polygon draw: On (P)" : "Polygon draw: Off (P)";
            polygonDrawToggle.textContent = label;
            polygonDrawToggle.ariaPressed = normalized ? "true" : "false";
            polygonDrawToggle.disabled = datasetType !== "seg";
        }
        if (!normalized) {
            polygonDraft = null;
            polygonDrag = null;
        }
        if (!silent && datasetType === "seg") {
            const msg = normalized ? "Polygon drawing enabled (click to add points, double-click to close, Esc to cancel)." : "Polygon drawing paused; click existing polygons to select/move. Press P to toggle back on.";
            setSamStatus(msg, { variant: "info", duration: 3000 });
        }
    };

    function refreshPolygonDetailVisibility() {
        if (!polygonSimplifyField) return;
        const show = datasetType === "seg" && samMode;
        polygonSimplifyField.style.display = show ? "" : "none";
    }

    const setDatasetType = (nextType) => {
        const normalized = nextType === "seg" ? "seg" : "bbox";
        datasetType = normalized;
        // Clear polygon draft when switching to bbox
        if (datasetType === "bbox") {
            polygonDraft = null;
            polygonDrag = null;
            setPolygonDrawEnabled(false, { silent: true });
        } else if (datasetType === "seg") {
            setPolygonDrawEnabled(true, { silent: true });
        }
        applyDatasetModeConstraints();
        if (!datasetTypeBadge) {
            datasetTypeBadge = document.getElementById("datasetTypeBadge");
        }
        if (datasetTypeBadge) {
            const label = normalized === "seg" ? "Polygon / YOLO-seg mode" : "BBox mode";
            datasetTypeBadge.textContent = `Dataset mode: ${label}`;
            datasetTypeBadge.title =
                normalized === "seg"
                    ? "Polygon mode: click to add points, double-click to close, drag vertices to edit."
                    : "BBox mode";
        }
        refreshPolygonDetailVisibility();
    };

    function applyDatasetModeConstraints() {
        const isSeg = datasetType === "seg";
        // Keep auto class available in both modes; segmentation users may still want predicted classes for bboxes.
        if (samModeCheckbox) {
            samModeCheckbox.disabled = false; // still allow SAM text prompts in seg mode
        }
        if (pointModeCheckbox) {
            pointModeCheckbox.disabled = !samMode;
            if (!samMode) {
                pointModeCheckbox.checked = false;
                updatePointModeState(false);
            }
        }
        if (multiPointModeCheckbox) {
            multiPointModeCheckbox.disabled = !samMode;
            if (!samMode) {
                multiPointModeCheckbox.checked = false;
                updateMultiPointModeState(false);
            }
        }
        if (polygonDrawToggle) {
            polygonDrawToggle.disabled = !isSeg;
            if (!isSeg) {
                setPolygonDrawEnabled(false, { silent: true });
            }
        }
        if (samStatusEl) {
            const modeNote = isSeg
                ? "Polygon mode: click to add vertices, double-click to close. Bbox-only tools are disabled."
                : "BBox mode: drag to create boxes; enable SAM/Auto for tweaks.";
            samStatusEl.dataset.modeNote = modeNote;
        }
    }

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
            datasetTypeBadge = document.getElementById("datasetTypeBadge");
            setDatasetType(datasetType);
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
        const isSegDataset = datasetType === "seg";
        const segBboxMode = isSegDataset && !polygonDrawEnabled;
        if (isSegDataset && polygonDrawEnabled && !samMode) {
            drawNewPolygon(context);
            return;
        }
        const canPreview =
            mouse.buttonL === true &&
            currentClass !== null &&
            (currentBbox === null || isSegDataset) &&
            (segBboxMode || datasetType !== "seg" || samMode);
        if (canPreview) {
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

    const drawNewPolygon = (context) => {
        if (!polygonDraft || !Array.isArray(polygonDraft.points) || polygonDraft.points.length === 0) {
            return;
        }
        const strokeColor = getColorFromClass(polygonDraft.className || currentClass || "");
        const fillColor = withAlpha(strokeColor, 0.2);
        context.save();
        context.strokeStyle = strokeColor;
        context.fillStyle = fillColor;
        context.lineWidth = Math.max(1, 1.2 * scale);
        context.beginPath();
        polygonDraft.points.forEach((pt, idx) => {
            const x = zoomX(pt.x);
            const y = zoomY(pt.y);
            if (idx === 0) {
                context.moveTo(x, y);
            } else {
                context.lineTo(x, y);
            }
        });
        context.stroke();
        polygonDraft.points.forEach((pt) => {
            const x = zoomX(pt.x);
            const y = zoomY(pt.y);
            context.beginPath();
            context.arc(x, y, Math.max(3, 4 * scale), 0, Math.PI * 2);
            context.fill();
        });
        context.restore();
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
                const pulseNow = performance.now();
                const pulseAlpha = 0.35 + 0.25 * Math.sin(pulseNow / 200);
                const highlightColor = `rgba(255, 213, 79, ${Math.min(0.9, 0.6 + pulseAlpha)})`;
                const highlightDash = [6 * scale, 4 * scale];

                context.font = context.font.replace(/\d+px/, `${Math.max(8, zoom(fontBaseSize))}px`);
                context.fillStyle = strokeColor;
                context.fillText(className, zoomX(bbox.x), zoomY(bbox.y - 2));

                context.setLineDash([]);
                context.lineWidth = lineWidth;
                context.strokeStyle = strokeColor;
                context.fillStyle = fillColor;
                const isPolygon = bbox.type === "polygon" || (Array.isArray(bbox.points) && bbox.points.length >= 3);
                if (isPolygon) {
                    context.beginPath();
                    bbox.points.forEach((pt, idx) => {
                        const x = zoomX(pt.x);
                        const y = zoomY(pt.y);
                        if (idx === 0) {
                            context.moveTo(x, y);
                        } else {
                            context.lineTo(x, y);
                        }
                    });
                    context.closePath();
                    context.fill();
                    context.stroke();
                    if (isCurrent) {
                        bbox.points.forEach((pt, idx) => {
                            const x = pt.x;
                            const y = pt.y;
                            drawCornerHandle(context, x, y, strokeColor);
                            if (idx === 0) {
                                context.beginPath();
                                context.arc(zoomX(x), zoomY(y), Math.max(4, 5 * scale), 0, Math.PI * 2);
                                context.stroke();
                            }
                        });
                        context.save();
                        context.strokeStyle = highlightColor;
                        context.lineWidth = Math.max(2.5, 2.5 * scale);
                        context.setLineDash(highlightDash);
                        context.lineDashOffset = (pulseNow / 20) % (10 * scale);
                        context.beginPath();
                        bbox.points.forEach((pt, idx) => {
                            const x = zoomX(pt.x);
                            const y = zoomY(pt.y);
                            if (idx === 0) {
                                context.moveTo(x, y);
                            } else {
                                context.lineTo(x, y);
                            }
                        });
                        context.closePath();
                        context.stroke();
                        context.restore();
                    }
                } else {
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
                    if (isCurrent) {
                        context.save();
                        context.strokeStyle = highlightColor;
                        context.lineWidth = Math.max(2.5, 2.5 * scale);
                        context.setLineDash(highlightDash);
                        context.lineDashOffset = (pulseNow / 20) % (10 * scale);
                        context.strokeRect(
                            zoomX(bbox.x),
                            zoomY(bbox.y),
                            zoom(bbox.width),
                            zoom(bbox.height)
                        );
                        context.restore();
                    }
                    if (bbox.marked === true) {
                        setBboxCoordinates(bbox.x, bbox.y, bbox.width, bbox.height);
                    }
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

    function isPointInsideBbox(bbox, x, y) {
        if (!bbox) return false;
        const x1 = bbox.x;
        const y1 = bbox.y;
        const x2 = bbox.x + bbox.width;
        const y2 = bbox.y + bbox.height;
        return x >= x1 && x <= x2 && y >= y1 && y <= y2;
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
                const segBboxMode = datasetType === "seg" && !polygonDrawEnabled;
                if (datasetType !== "seg" || segBboxMode) {
                    const insideExisting = currentBbox && isPointInsideBbox(currentBbox.bbox, mouse.realX, mouse.realY);
                    if (!insideExisting) {
                        currentBbox = null;
                    }
                }
            }
        }
        const isSegDataset = datasetType === "seg";
        if (isSegDataset) {
            const handled = await handlePolygonPointer(event, oldRealX, oldRealY);
            if (handled) {
                if (event.type === "mouseup" || event.type === "mouseout") {
                    mouse.buttonR = false;
                    mouse.buttonL = false;
                }
                return;
            }
            // If polygon drawing is off, allow bbox flows to continue in seg mode.
            if (!samMode && polygonDrawEnabled) {
                if (event.type === "mouseup" || event.type === "mouseout") {
                    mouse.buttonR = false;
                    mouse.buttonL = false;
                }
                return;
            }
        }
        if (event.type === "mouseup" || event.type === "mouseout") {
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
                        await samPointAutoPrompt(mouse.realX, mouse.realY);
                    } else {
                        await samPointPrompt(mouse.realX, mouse.realY);
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
                                await samBboxAutoPrompt(currentBbox.bbox);
                            }
                            else if (autoMode) {
                                await autoPredictNewCrop(currentBbox.bbox);
                            }
                            else if (samMode) {
                                await samBboxPrompt(currentBbox.bbox);
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

        // In seg+SAM mode, allow bbox flows (preview, pan) to run; in plain seg mode we already returned above.
        if (datasetType === "seg" && samMode) {
            const polygonSelected = currentBbox && (currentBbox.bbox?.type === "polygon" || (Array.isArray(currentBbox.bbox?.points) && currentBbox.bbox.points.length >= 3));
            if (!polygonSelected) {
                moveBbox();
                resizeBbox();
            }
            changeCursorByLocation();
            panImage(oldRealX, oldRealY);
            return;
        } else if (datasetType === "seg" && polygonDrawEnabled) {
            return;
        }
        moveBbox();
        resizeBbox();
        changeCursorByLocation();
        panImage(oldRealX, oldRealY);
    }

    function pointInPolygon(x, y, points) {
        if (!Array.isArray(points) || points.length < 3) return false;
        let inside = false;
        for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
            const xi = points[i].x, yi = points[i].y;
            const xj = points[j].x, yj = points[j].y;
            const intersect = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    function findPolygonAt(x, y) {
        if (!currentImage || !bboxes[currentImage.name]) return null;
        const imgBxs = bboxes[currentImage.name];
        let found = null;
        Object.keys(imgBxs).forEach((className) => {
            imgBxs[className].forEach((ann, idx) => {
                if (found || ann.type !== "polygon" || !Array.isArray(ann.points)) return;
                const distThreshold = Math.max(6, 10 / scale);
                // Prefer vertex hit
                ann.points.forEach((pt, vIdx) => {
                    const dx = pt.x - x;
                    const dy = pt.y - y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist <= distThreshold && !found) {
                        found = { bbox: ann, className, index: idx, vertexIndex: vIdx };
                    }
                });
                if (!found && pointInPolygon(x, y, ann.points)) {
                    found = { bbox: ann, className, index: idx, vertexIndex: null };
                }
            });
        });
        return found;
    }

    function clampPointToImage(pt) {
        if (!currentImage) return pt;
        const w = currentImage.width || currentImage.object?.naturalWidth || 0;
        const h = currentImage.height || currentImage.object?.naturalHeight || 0;
        return {
            x: Math.max(0, Math.min(w, pt.x)),
            y: Math.max(0, Math.min(h, pt.y)),
        };
    }

    function finalizePolygonDraft() {
        if (!polygonDraft || !currentImage || !currentClass) {
            polygonDraft = null;
            return;
        }
        const pts = (polygonDraft.points || []).map(clampPointToImage);
        if (pts.length < 3) {
            polygonDraft = null;
            return;
        }
        const xs = pts.map((p) => p.x);
        const ys = pts.map((p) => p.y);
        const minX = Math.max(0, Math.min(...xs));
        const maxX = Math.min(currentImage.width, Math.max(...xs));
        const minY = Math.max(0, Math.min(...ys));
        const maxY = Math.min(currentImage.height, Math.max(...ys));
        const bboxRecord = {
            type: "polygon",
            points: pts,
            x: minX,
            y: minY,
            width: Math.max(0, maxX - minX),
            height: Math.max(0, maxY - minY),
            marked: false,
            class: polygonDraft.className || currentClass,
        };
        stampBboxCreation(bboxRecord);
        if (!bboxes[currentImage.name]) {
            bboxes[currentImage.name] = {};
        }
        const targetClass = polygonDraft.className || currentClass;
        if (!bboxes[currentImage.name][targetClass]) {
            bboxes[currentImage.name][targetClass] = [];
        }
        // Deduplicate exact same polygon for the same class
        const existing = bboxes[currentImage.name][targetClass].some((ann) => {
            if (ann.type !== "polygon" || !Array.isArray(ann.points) || ann.points.length !== pts.length) return false;
            return ann.points.every((pt, idx) => Math.abs(pt.x - pts[idx].x) < 1e-3 && Math.abs(pt.y - pts[idx].y) < 1e-3);
        });
        if (!existing) {
            bboxes[currentImage.name][targetClass].push(bboxRecord);
            currentBbox = {
                bbox: bboxRecord,
                index: bboxes[currentImage.name][targetClass].length - 1,
                originalX: bboxRecord.x,
                originalY: bboxRecord.y,
                originalWidth: bboxRecord.width,
                originalHeight: bboxRecord.height,
                moving: false,
                resizing: null,
            };
        } else {
            currentBbox = null;
        }
        setDatasetType("seg");
        polygonDraft = null;
    }

    async function handlePolygonPointer(event, prevX, prevY) {
        if (!currentClass || !currentImage) {
            return false;
        }
        const samActive = samMode === true;
        if (mouse.buttonR) {
            panImage(prevX, prevY);
        }
        if (event.type === "mousedown") {
            if (event.which === 3) {
                // Right click: cancel or finalize draft
                if (!samActive && polygonDraft && polygonDraft.points.length >= 3) {
                    finalizePolygonDraft();
                } else if (!samActive) {
                    polygonDraft = null;
                }
                polygonDrag = null;
                currentBbox = null;
                return true;
            }
            if (event.which === 1) {
                const hit = findPolygonAt(mouse.realX, mouse.realY);
                if (hit) {
                    currentClass = hit.className || currentClass;
                    currentBbox = {
                        bbox: hit.bbox,
                        index: hit.index,
                        originalX: hit.bbox.x,
                        originalY: hit.bbox.y,
                        originalWidth: hit.bbox.width,
                        originalHeight: hit.bbox.height,
                        moving: false,
                        resizing: null,
                    };
                    if (!samActive && hit.vertexIndex !== null && hit.vertexIndex !== undefined) {
                        polygonDrag = { ...hit, vertexIndex: hit.vertexIndex };
                    } else {
                        polygonDrag = { ...hit, vertexIndex: null };
                    }
                    return true;
                }
                if (samActive) {
                    // In SAM mode, do not start polygon drafting; allow bbox drawing.
                    currentBbox = null;
                    polygonDrag = null;
                    polygonDraft = null;
                    return false;
                }
                if (!polygonDrawEnabled) {
                    polygonDraft = null;
                    polygonDrag = null;
                    currentBbox = null;
                    return false;
                }
                // Double click closes polygon
                if (polygonDraft && event.detail > 1 && polygonDraft.points.length >= 3) {
                    finalizePolygonDraft();
                    return true;
                }
                if (!polygonDraft) {
                    polygonDraft = { className: currentClass, points: [] };
                }
                polygonDraft.points.push({ x: mouse.realX, y: mouse.realY });
            }
        } else if (event.type === "mousemove") {
            if (polygonDrag && polygonDrag.vertexIndex !== null && polygonDrag.vertexIndex !== undefined) {
                const pts = polygonDrag.bbox.points;
                pts[polygonDrag.vertexIndex] = clampPointToImage({ x: mouse.realX, y: mouse.realY });
                const xs = pts.map((p) => p.x);
                const ys = pts.map((p) => p.y);
                polygonDrag.bbox.x = Math.min(...xs);
                polygonDrag.bbox.y = Math.min(...ys);
                polygonDrag.bbox.width = Math.max(...xs) - polygonDrag.bbox.x;
                polygonDrag.bbox.height = Math.max(...ys) - polygonDrag.bbox.y;
            }
        } else if (event.type === "mouseup" || event.type === "mouseout") {
            polygonDrag = null;
        }
        return false;
    }

    const storeNewBbox = (movedWidth, movedHeight) => {
        const bbox = {
            type: "bbox",
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
        if (datasetType === "seg" && polygonDrawEnabled) {
            if (currentBbox && currentBbox.bbox) {
                currentBbox.bbox.marked = true;
            }
            return;
        }
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
        if (datasetType === "seg") {
            setSamStatus("Tweak is only available in bbox mode.", { variant: "warn", duration: 3000 });
            return false;
        }
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
                await samBboxAutoPrompt(targetBbox);
            } else if (samMode) {
                await samBboxPrompt(targetBbox);
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
        if (datasetType === "seg") {
            setSamStatus("Batch tweak is only available in bbox mode.", { variant: "warn", duration: 3000 });
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
                            updateSam3ClassOptions({ resetOverride: true });
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
        updateSam3ClassOptions({ resetOverride: true });
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
        updateSam3ClassOptions({ preserveSelection: true });
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
        setDatasetType("bbox");
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
                            const cols = rows[i].trim().split(/\s+/).filter(Boolean);
                            if (cols.length < 5) continue;
                            const clsIdx = parseInt(cols[0], 10);
                            let className = null;
                            for (const name in classes) {
                                if (classes[name] === clsIdx) {
                                    className = name;
                                    break;
                                }
                            }
                            if (!className) continue;
                            if (typeof bbox[className] === "undefined") {
                                bbox[className] = [];
                            }
                            // YOLO-seg: class + polygon coords (x y ...), YOLO-bbox: class cx cy w h
                            if (cols.length >= 7) {
                                const pts = [];
                                for (let j = 1; j + 1 < cols.length; j += 2) {
                                    const px = parseFloat(cols[j]) * image.width;
                                    const py = parseFloat(cols[j + 1]) * image.height;
                                    if (Number.isFinite(px) && Number.isFinite(py)) {
                                        pts.push({ x: px, y: py });
                                    }
                                }
                                if (pts.length >= 3) {
                                    const xs = pts.map((p) => p.x);
                                    const ys = pts.map((p) => p.y);
                                    const minX = Math.max(0, Math.min(...xs));
                                    const maxX = Math.min(image.width, Math.max(...xs));
                                    const minY = Math.max(0, Math.min(...ys));
                                    const maxY = Math.min(image.height, Math.max(...ys));
                                    const bboxRecord = {
                                        type: "polygon",
                                        points: pts,
                                        x: minX,
                                        y: minY,
                                        width: Math.max(0, maxX - minX),
                                        height: Math.max(0, maxY - minY),
                                        marked: false,
                                        class: className
                                    };
                                    stampBboxCreation(bboxRecord);
                                    bbox[className].push(bboxRecord);
                                    setDatasetType("seg");
                                    noteImportedBbox();
                                    continue;
                                }
                            }
                            // Fallback to bbox
                            const cx = parseFloat(cols[1]);
                            const cy = parseFloat(cols[2]);
                            const wNorm = parseFloat(cols[3]);
                            const hNorm = parseFloat(cols[4]);
                            const width = wNorm * image.width;
                            const x = cx * image.width - width * 0.5;
                            const height = hNorm * image.height;
                            const y = cy * image.height - height * 0.5;
                            const bboxRecord = {
                                type: "bbox",
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
                                        type: "bbox",
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
                            type: "bbox",
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

    function summarizeGeometry() {
        let polygonCount = 0;
        let bboxCount = 0;
        Object.values(bboxes).forEach((classBuckets) => {
            Object.values(classBuckets || {}).forEach((items) => {
                (items || []).forEach((ann) => {
                    if (ann && ann.type === "polygon") {
                        polygonCount += 1;
                    } else {
                        bboxCount += 1;
                    }
                });
            });
        });
        return { polygonCount, bboxCount };
    }

    function validateGeometryForSave() {
        const { polygonCount, bboxCount } = summarizeGeometry();
        if (datasetType === "seg" && bboxCount > 0) {
            return { ok: false, message: "This dataset is in polygon mode but contains bboxes. Remove or convert them before saving." };
        }
        if (datasetType === "bbox" && polygonCount > 0) {
            return { ok: false, message: "This dataset is in bbox mode but contains polygons. Switch to seg mode or remove polygons before saving." };
        }
        return { ok: true, polygonCount, bboxCount };
    }

    const listenBboxSave = () => {
        document.getElementById("saveBboxes").addEventListener("click", () => {
            const validation = validateGeometryForSave();
            if (!validation.ok) {
                alert(validation.message);
                return;
            }
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
                        const classIdx = classes[className];
                        if (datasetType === "seg" && Array.isArray(bbox.points) && bbox.points.length >= 3) {
                            const coords = bbox.points
                                .map((pt) => {
                                    const nx = pt.x / image.width;
                                    const ny = pt.y / image.height;
                                    return `${nx} ${ny}`;
                                })
                                .join(" ");
                            result.push(`${classIdx} ${coords}`);
                        } else {
                            const x = (bbox.x + bbox.width / 2) / image.width;
                            const y = (bbox.y + bbox.height / 2) / image.height;
                            const w = bbox.width / image.width;
                            const h = bbox.height / image.height;
                            result.push(`${classIdx} ${x} ${y} ${w} ${h}`);
                        }
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

            if (datasetType === "seg" && (key === 27 || event.key === "Escape")) {
                polygonDraft = null;
                polygonDrag = null;
                currentBbox = null;
                event.preventDefault();
                return;
            }

            if (!event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey && (key === 80 || event.key === "p" || event.key === "P")) {
                if (datasetType !== "seg") {
                    setDatasetType("seg");
                    setPolygonDrawEnabled(true);
                    event.preventDefault();
                    return;
                }
                setPolygonDrawEnabled(!polygonDrawEnabled);
                event.preventDefault();
                return;
            }

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
        if (datasetType === "seg") {
          alert("Crop & Save is only available for bbox datasets.");
          return;
        }
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

    async function loadSam3RecipePresets() {
        if (!sam3RecipeElements.presetSelect) return;
        try {
            const resp = await fetch(`${API_ROOT}/agent_mining/recipes`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            sam3RecipeElements.presetSelect.innerHTML = "";
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "Select recipe…";
            sam3RecipeElements.presetSelect.appendChild(placeholder);
            (Array.isArray(data) ? data : []).forEach((p) => {
                const opt = document.createElement("option");
                opt.value = p.id || p._path || "";
                const cls = p.class_name ? ` • ${p.class_name}` : "";
                opt.textContent = `${p.label || p.id || opt.value}${cls}`;
                sam3RecipeElements.presetSelect.appendChild(opt);
            });
        } catch (err) {
            console.error("Load recipe presets failed", err);
            setSam3RecipeStatus("Failed to load recipe presets.", "warn");
        }
    }

    async function saveSam3RecipePreset() {
        const recipe = sam3RecipeState.recipe;
        if (!recipe || !recipe.steps || !recipe.steps.length) {
            setSam3RecipeStatus("Load a recipe first, then save.", "warn");
            return;
        }
        try {
            const payload = {
                label: sam3RecipeElements.presetNameInput?.value || recipe.label || "",
                class_name: recipe.class_name,
                class_id: recipe.class_id,
                recipe: { steps: recipe.steps, summary: recipe.summary },
                dataset_id: null,
            };
            const resp = await fetch(`${API_ROOT}/agent_mining/recipes`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const detail = await resp.text();
                throw new Error(detail || `HTTP ${resp.status}`);
            }
            await loadSam3RecipePresets();
            setSam3RecipeStatus("Saved recipe preset.", "success");
        } catch (err) {
            console.error("Save recipe preset failed", err);
            setSam3RecipeStatus(err.message || "Save failed.", "error");
        }
    }

    async function loadSam3RecipePreset() {
        const presetId = sam3RecipeElements.presetSelect?.value;
        if (!presetId) {
            setSam3RecipeStatus("Choose a recipe preset to load.", "warn");
            return;
        }
        try {
            const resp = await fetch(`${API_ROOT}/agent_mining/recipes/${encodeURIComponent(presetId)}`);
            if (!resp.ok) {
                const detail = await resp.text();
                throw new Error(detail || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const parsed = {
                label: data.label || data.id || (data.recipe && (data.recipe.label || data.recipe.id)) || "",
                class_name: data.class_name || (data.recipe && data.recipe.class_name),
                class_id: data.class_id ?? (data.recipe && data.recipe.class_id),
                steps: Array.isArray(data.recipe?.steps)
                    ? data.recipe.steps
                    : Array.isArray(data.steps)
                        ? data.steps
                              .map((s) => ({
                                  prompt: typeof s.prompt === "string" ? s.prompt.trim() : "",
                                  threshold: typeof s.threshold === "number" ? s.threshold : null,
                              }))
                          .filter((s) => s.prompt && s.threshold !== null)
                    : [],
            };
            if (!parsed.steps.length) throw new Error("Preset has no steps.");
            // Validate class exists in label map.
            const classNames = orderedClassNames();
            const lowerToName = new Map(classNames.map((n) => [n.toLowerCase(), n]));
            const target = parsed.class_name || (typeof parsed.class_id === "number" ? classNames[parsed.class_id] : null);
            if (!target || !lowerToName.has(target.toLowerCase())) {
                throw new Error(`Class not in label map: ${parsed.class_name || parsed.class_id || ""}`);
            }
            parsed.class_name = lowerToName.get(target.toLowerCase());
            sam3RecipeState.recipe = parsed;
            if (sam3RecipeElements.applyButton) sam3RecipeElements.applyButton.disabled = false;
            if (sam3RecipeElements.presetNameInput) sam3RecipeElements.presetNameInput.value = parsed.label || parsed.class_name;
            setSam3RecipeStatus(`Loaded preset for ${parsed.class_name} (${parsed.steps.length} steps).`, "success");
        } catch (err) {
            console.error("Load recipe preset failed", err);
            setSam3RecipeStatus(err.message || "Load failed.", "error");
            sam3RecipeState.recipe = null;
            if (sam3RecipeElements.applyButton) sam3RecipeElements.applyButton.disabled = true;
        }
    }

    async function deleteSam3RecipePreset() {
        const presetId = sam3RecipeElements.presetSelect?.value;
        if (!presetId) {
            setSam3RecipeStatus("Choose a recipe preset to delete.", "warn");
            return;
        }
        const confirmed = window.confirm("Delete this recipe? This cannot be undone.");
        if (!confirmed) return;
        try {
            const resp = await fetch(`${API_ROOT}/agent_mining/recipes/${encodeURIComponent(presetId)}`, {
                method: "DELETE",
            });
            if (!resp.ok) {
                const detail = await resp.text();
                throw new Error(detail || `HTTP ${resp.status}`);
            }
            sam3RecipeState.recipe = null;
            if (sam3RecipeElements.applyButton) sam3RecipeElements.applyButton.disabled = true;
            if (sam3RecipeElements.presetNameInput) sam3RecipeElements.presetNameInput.value = "";
            await loadSam3RecipePresets();
            setSam3RecipeStatus("Deleted recipe preset.", "success");
        } catch (err) {
            console.error("Delete recipe preset failed", err);
            setSam3RecipeStatus(err.message || "Delete failed.", "error");
        }
    }

})();
