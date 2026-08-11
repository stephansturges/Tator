(function (root, factory) {
    const api = factory(root);
    if (typeof module === "object" && module.exports) {
        module.exports = api;
    }
    root.ClassSplitGraphView = api;
    if (root.document) {
        const start = () => api.init();
        if (root.document.readyState === "loading") {
            root.document.addEventListener("DOMContentLoaded", start, { once: true });
        } else {
            start();
        }
    }
}(typeof globalThis !== "undefined" ? globalThis : this, function (root) {
    "use strict";

    const DEFAULTS = Object.freeze({
        sizePercent: 100,
        opacityPercent: 100,
        labelDensityPercent: 0,
    });
    const stateByGraph = new WeakMap();
    let initialized = false;

    function clamp(value, minimum, maximum, fallback) {
        const numeric = Number(value);
        return Number.isFinite(numeric)
            ? Math.max(minimum, Math.min(maximum, numeric))
            : fallback;
    }

    function settingsFromDocument(doc = root.document) {
        const classColorMode = String(
            doc?.getElementById("classSplitColorMode")?.value || "class"
        ) === "class";
        return {
            sizePercent: clamp(doc?.getElementById("classSplitMarkerSize")?.value, 50, 250, 100),
            opacityPercent: clamp(doc?.getElementById("classSplitMarkerOpacity")?.value, 20, 100, 100),
            labelDensityPercent: classColorMode
                ? clamp(doc?.getElementById("classSplitLabelDensity")?.value, 0, 100, 0)
                : 0,
        };
    }

    function cloneValue(value) {
        return Array.isArray(value) ? value.slice() : value;
    }

    function markerTrace(trace) {
        return !!(
            trace
            && String(trace.mode || "").split("+").includes("markers")
            && trace.marker
            && Array.isArray(trace.customdata)
        );
    }

    function classLabelTrace(trace) {
        const name = String(trace?.name || "").trim();
        return markerTrace(trace)
            && trace.showlegend !== false
            && name
            && !["Objects", "Likely wrong class", "Selected object"].includes(name);
    }

    function captureBaseline(trace) {
        return {
            customdata: Array.isArray(trace?.customdata) ? trace.customdata.slice() : [],
            size: cloneValue(trace?.marker?.size),
            opacity: cloneValue(trace?.marker?.opacity),
            mode: String(trace?.mode || "markers"),
            texttemplate: cloneValue(trace?.texttemplate),
            textposition: cloneValue(trace?.textposition),
        };
    }

    function scaleValue(value, factor, fallback) {
        if (Array.isArray(value)) {
            return value.map((item) => {
                const numeric = Number(item);
                return Number.isFinite(numeric) ? numeric * factor : fallback;
            });
        }
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric * factor : fallback;
    }

    function stableHash(value) {
        const text = String(value || "");
        let hash = 2166136261;
        for (let index = 0; index < text.length; index += 1) {
            hash ^= text.charCodeAt(index);
            hash = Math.imul(hash, 16777619);
        }
        return hash >>> 0;
    }

    function sampledLabels(trace, densityPercent) {
        const ids = Array.isArray(trace?.customdata) ? trace.customdata : [];
        const labels = new Array(ids.length).fill("");
        if (!classLabelTrace(trace) || densityPercent <= 0 || !ids.length) {
            return labels;
        }
        const count = Math.min(
            ids.length,
            Math.max(1, Math.round(ids.length * densityPercent / 100)),
        );
        ids.map((id, index) => ({ index, hash: stableHash(id) }))
            .sort((left, right) => left.hash - right.hash || left.index - right.index)
            .slice(0, count)
            .forEach(({ index }) => {
                labels[index] = String(trace.name || "");
            });
        return labels;
    }

    function captureAndApply(graph, settings = settingsFromDocument()) {
        if (!graph || !Array.isArray(graph.data)) {
            return Promise.resolve(false);
        }
        const state = stateByGraph.get(graph) || {};
        state.baselines = graph.data.map(captureBaseline);
        state.pendingSettings = null;
        stateByGraph.set(graph, state);
        return applyToGraph(graph, settings);
    }

    function applyToGraph(graph, settings = settingsFromDocument()) {
        const plotly = root.Plotly;
        if (!plotly?.restyle || !graph || !Array.isArray(graph.data)) {
            return Promise.resolve(false);
        }
        let state = stateByGraph.get(graph);
        if (!state || state.baselines.length !== graph.data.length) {
            state = {
                baselines: graph.data.map(captureBaseline),
                pendingSettings: null,
                applyPromise: null,
            };
            stateByGraph.set(graph, state);
        }
        state.pendingSettings = {
            sizePercent: clamp(settings.sizePercent, 50, 250, DEFAULTS.sizePercent),
            opacityPercent: clamp(settings.opacityPercent, 20, 100, DEFAULTS.opacityPercent),
            labelDensityPercent: clamp(settings.labelDensityPercent, 0, 100, DEFAULTS.labelDensityPercent),
        };
        if (state.applyPromise) {
            return state.applyPromise;
        }
        state.applyPromise = (async () => {
            let applied = false;
            while (state.pendingSettings) {
                const resolved = state.pendingSettings;
                state.pendingSettings = null;
                const traceIndexes = [];
                const sizes = [];
                const opacities = [];
                const modes = [];
                const templates = [];
                const positions = [];
                graph.data.forEach((trace, traceIndex) => {
                    if (!markerTrace(trace)) return;
                    const baseline = state.baselines[traceIndex] || captureBaseline(trace);
                    const labels = sampledLabels(trace, resolved.labelDensityPercent);
                    const showLabels = labels.some(Boolean);
                    traceIndexes.push(traceIndex);
                    sizes.push(scaleValue(baseline.size, resolved.sizePercent / 100, 8));
                    opacities.push(scaleValue(
                        baseline.opacity,
                        resolved.opacityPercent / 100,
                        resolved.opacityPercent / 100,
                    ));
                    modes.push(
                        showLabels && !baseline.mode.split("+").includes("text")
                            ? `${baseline.mode}+text`
                            : baseline.mode
                    );
                    templates.push(showLabels ? labels : baseline.texttemplate ?? null);
                    positions.push(showLabels ? "top center" : baseline.textposition ?? "top center");
                });
                if (traceIndexes.length) {
                    await plotly.restyle(graph, {
                        "marker.size": sizes,
                        "marker.opacity": opacities,
                        mode: modes,
                        texttemplate: templates,
                        textposition: positions,
                    }, traceIndexes);
                    applied = true;
                }
            }
            return applied;
        })().finally(() => {
            state.applyPromise = null;
        });
        return state.applyPromise;
    }

    function remapArrayById(oldIds, oldValue, newIds) {
        if (!Array.isArray(oldValue)) return cloneValue(oldValue);
        const byId = new Map(
            oldIds.map((id, index) => [String(id || ""), oldValue[index]])
        );
        return newIds.map((id) => byId.get(String(id || "")));
    }

    function syncAfterExternalRestyle(graph, settings = settingsFromDocument()) {
        if (!graph || !Array.isArray(graph.data)) {
            return Promise.resolve(false);
        }
        const previous = stateByGraph.get(graph)?.baselines || [];
        const baselines = graph.data.map((trace, traceIndex) => {
            const old = previous[traceIndex];
            if (!old || !Array.isArray(old.size) || !Array.isArray(old.customdata)) {
                return old || captureBaseline(trace);
            }
            const sizeById = new Map(
                old.customdata.map((id, index) => [String(id || ""), old.size[index]])
            );
            return {
                ...old,
                customdata: trace.customdata.slice(),
                size: trace.customdata.map((id) => sizeById.get(String(id || ""))),
                opacity: remapArrayById(
                    old.customdata,
                    old.opacity,
                    trace.customdata,
                ),
                texttemplate: remapArrayById(
                    old.customdata,
                    old.texttemplate,
                    trace.customdata,
                ),
                textposition: remapArrayById(
                    old.customdata,
                    old.textposition,
                    trace.customdata,
                ),
            };
        });
        stateByGraph.set(graph, { baselines });
        return applyToGraph(graph, settings);
    }

    function updateOutputs(doc = root.document) {
        const settings = settingsFromDocument(doc);
        const values = {
            classSplitMarkerSizeValue: `${Math.round(settings.sizePercent)}%`,
            classSplitMarkerOpacityValue: `${Math.round(settings.opacityPercent)}%`,
            classSplitLabelDensityValue: String(
                doc?.getElementById("classSplitColorMode")?.value || "class"
            ) === "class"
                ? `${Math.round(settings.labelDensityPercent)}%`
                : "Off",
        };
        Object.entries(values).forEach(([id, value]) => {
            const output = doc?.getElementById(id);
            if (output) output.textContent = value;
        });
        const density = doc?.getElementById("classSplitLabelDensity");
        const colorMode = doc?.getElementById("classSplitColorMode");
        if (density) {
            density.disabled = !!colorMode
                && String(colorMode.value || "class") !== "class";
        }
        return settings;
    }

    function init(doc = root.document) {
        if (initialized || !doc) return;
        initialized = true;
        [
            "classSplitMarkerSize",
            "classSplitMarkerOpacity",
            "classSplitLabelDensity",
        ].forEach((id) => {
            doc.getElementById(id)?.addEventListener("input", () => {
                const settings = updateOutputs(doc);
                applyToGraph(
                    doc.getElementById("classSplitGraph"),
                    settings,
                ).catch((error) => {
                    root.console?.warn(
                        "Data Quality Explorer view control failed",
                        error,
                    );
                });
            });
        });
        doc.getElementById("classSplitColorMode")?.addEventListener(
            "change",
            () => {
                const settings = updateOutputs(doc);
                applyToGraph(
                    doc.getElementById("classSplitGraph"),
                    settings,
                ).catch((error) => {
                    root.console?.warn(
                        "Data Quality Explorer view control failed",
                        error,
                    );
                });
            },
        );
        updateOutputs(doc);
    }

    return {
        DEFAULTS,
        applyToGraph,
        captureAndApply,
        init,
        sampledLabels,
        settingsFromDocument,
        syncAfterExternalRestyle,
    };
}));
