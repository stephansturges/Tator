(function (root) {
    "use strict";

    let requestInFlight = false;
    let latestStatus = null;

    function apiRoot() {
        const configured = typeof root.getTatorApiRoot === "function"
            ? root.getTatorApiRoot()
            : root.location?.origin;
        return String(configured || "").replace(/\/$/, "");
    }

    function formatBytes(value) {
        const bytes = Math.max(0, Number(value) || 0);
        if (bytes < 1024) return `${Math.round(bytes)} B`;
        const units = ["KiB", "MiB", "GiB", "TiB"];
        let amount = bytes / 1024;
        let unit = units[0];
        for (
            let index = 1;
            index < units.length && amount >= 1024;
            index += 1
        ) {
            amount /= 1024;
            unit = units[index];
        }
        return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${unit}`;
    }

    function setMessage(message, tone = "") {
        const status = root.document?.getElementById(
            "classAnalysisCacheStatus"
        );
        if (!status) return;
        status.textContent = message;
        status.dataset.tone = tone;
    }

    function render(payload) {
        latestStatus = payload;
        const categories = payload?.categories || {};
        const total = formatBytes(payload?.total_bytes);
        const managed = formatBytes(
            payload?.managed_bytes ?? payload?.purgeable_bytes
        );
        const protectedBytes = formatBytes(
            payload?.protected_bytes
            ?? Math.max(
                0,
                Number(payload?.total_bytes || 0)
                - Number(payload?.purgeable_bytes || 0),
            )
        );
        const budget = Number(payload?.max_bytes) > 0
            ? ` / ${formatBytes(payload.max_bytes)}`
            : "";
        const crops = formatBytes(categories.image_packs?.bytes);
        const embeddings = formatBytes(
            categories.resume_embeddings?.bytes
        );
        const active = Array.isArray(payload?.active_users)
            ? payload.active_users
            : [];
        const suffix = active.length
            ? ` • ${active.length} active job${active.length === 1 ? "" : "s"}; clearing is locked`
            : "";
        const overBudget = Number(payload?.over_budget_bytes) > 0;
        setMessage(
            `${managed}${budget} regenerable • ${protectedBytes} protected • ${total} total • crops ${crops} • resumable embeddings ${embeddings}${suffix}`,
            active.length || overBudget ? "warn" : "",
        );
        const clear = root.document?.getElementById(
            "classAnalysisCacheClear"
        );
        if (clear) {
            clear.disabled = requestInFlight
                || active.length > 0
                || Number(payload?.purgeable_bytes) <= 0;
        }
    }

    async function parseResponse(response) {
        const text = await response.text();
        let payload = {};
        try {
            payload = text ? JSON.parse(text) : {};
        } catch (_error) {
            payload = {};
        }
        if (!response.ok) {
            const detail = payload?.detail;
            const message = typeof detail === "string"
                ? detail
                : detail?.message
                    || detail?.code
                    || `HTTP ${response.status}`;
            throw new Error(message);
        }
        return payload;
    }

    async function refresh() {
        if (requestInFlight) return;
        requestInFlight = true;
        let refreshedStatus = null;
        setMessage("Measuring cache usage ...");
        try {
            refreshedStatus = await parseResponse(
                await root.fetch(`${apiRoot()}/class_analysis/cache`)
            );
        } catch (error) {
            setMessage(
                `Cache usage unavailable: ${error.message || error}`,
                "error",
            );
        } finally {
            requestInFlight = false;
            if (refreshedStatus) render(refreshedStatus);
        }
    }

    async function clearRegenerableCaches() {
        if (requestInFlight) return;
        const amount = formatBytes(latestStatus?.purgeable_bytes);
        const confirmed = root.confirm(
            `Clear ${amount} of cached crops and resumable embeddings? `
            + "Source datasets, analysis results, review evidence, and "
            + "patch-reference banks are preserved."
        );
        if (!confirmed) return;
        requestInFlight = true;
        let reclaimed = "";
        let refreshedStatus = null;
        setMessage(
            "Clearing cached crops and resumable embeddings ..."
        );
        try {
            const payload = await parseResponse(
                await root.fetch(
                    `${apiRoot()}/class_analysis/cache/purge`,
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            categories: [
                                "image_packs",
                                "resume_embeddings",
                            ],
                        }),
                    },
                )
            );
            refreshedStatus = payload.after || {};
            reclaimed = formatBytes(payload.bytes_reclaimed);
        } catch (error) {
            setMessage(
                `Cache clear blocked: ${error.message || error}`,
                "error",
            );
        } finally {
            requestInFlight = false;
            if (refreshedStatus) render(refreshedStatus);
            if (reclaimed) {
                setMessage(
                    `Cleared ${reclaimed}. `
                    + (root.document.getElementById(
                        "classAnalysisCacheStatus"
                    )?.textContent || ""),
                    "success",
                );
            }
        }
    }

    function init() {
        root.document?.getElementById(
            "classAnalysisCacheRefresh"
        )?.addEventListener("click", refresh);
        root.document?.getElementById(
            "classAnalysisCacheClear"
        )?.addEventListener("click", clearRegenerableCaches);
        root.document?.getElementById(
            "tabClassSplitButton"
        )?.addEventListener("click", refresh);
        refresh();
    }

    if (root.document?.readyState === "loading") {
        root.document.addEventListener(
            "DOMContentLoaded",
            init,
            { once: true },
        );
    } else if (root.document) {
        init();
    }
}(typeof globalThis !== "undefined" ? globalThis : this));
