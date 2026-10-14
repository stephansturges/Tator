(function (root) {
    "use strict";
    const PREFIX = "tator.dqe.library.v1.";
    const LEGACY = "tator.dataQuality.session.v1";
    const safeRead = (storage, key, fallback = null) => {
        try { return JSON.parse(storage?.getItem(key) || "null") ?? fallback; }
        catch (_) { return fallback; }
    };
    function create() {
        function availableStorage(name) {
            try { return root[name]; }
            catch (_) { return { getItem: () => null, key: () => null, length: 0, removeItem() {}, setItem() { throw new Error("Browser recovery storage is unavailable."); } }; }
        }
        const localStorage = availableStorage("localStorage");
        const sessionStorage = availableStorage("sessionStorage");
        let backendId = "";
        const tabKey = `${PREFIX}tab`;
        // Duplicate Tab/window.open clone sessionStorage. A fresh document ID
        // keeps recovery writes isolated; predecessor records remain readable.
        const predecessorId = safeRead(sessionStorage, tabKey);
        const lineageKey = `${PREFIX}predecessors`;
        const savedLineage = safeRead(sessionStorage, lineageKey, []);
        const predecessorIds = Array.from(new Set([predecessorId, ...(Array.isArray(savedLineage) ? savedLineage : [])].filter((id) => typeof id === "string" && id)));
        const navigation = root.performance?.getEntriesByType?.("navigation")?.[0]?.type;
        const mayResumeLease = navigation === "reload" || navigation === "back_forward";
        const tabId = root.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        try {
            sessionStorage.setItem(tabKey, JSON.stringify(tabId));
            sessionStorage.setItem(lineageKey, JSON.stringify(predecessorIds));
        } catch (_) { /* Reads remain usable. */ }
        const key = (kind, id = "", owner = backendId) => `${PREFIX}${encodeURIComponent(owner)}.${kind}.${encodeURIComponent(id)}`;
        function write(storage, name, value) {
            const text = JSON.stringify(value);
            storage.setItem(name, text);
            if (storage.getItem(name) !== text) throw new Error("Browser recovery metadata could not be verified.");
            return value;
        }
        function requireBackend() {
            if (!backendId) throw new Error("The backend identity has not been verified.");
        }
        function readSession(jobId = "") {
            if (!backendId) return safeRead(sessionStorage, LEGACY, {});
            const active = readActive();
            const id = String(jobId || active.jobId || "draft");
            return {
                ...safeRead(localStorage, key("view", id), {}),
                ...readOwned(localStorage, "recovery", id),
            };
        }
        function readOwned(storage, kind, id) {
            const own = safeRead(storage, key(kind, `${id}:${tabId}`));
            if (own) return own;
            for (const previousId of predecessorIds) {
                const previousKey = key(kind, `${id}:${previousId}`);
                if (storage.getItem(previousKey) !== null) return { ...safeRead(storage, previousKey, {}) };
            }
            return {};
        }
        function readActive() {
            if (!backendId) return {};
            const own = safeRead(sessionStorage, key("active", tabId));
            if (own) return own;
            for (const previousId of predecessorIds) {
                const previous = safeRead(sessionStorage, key("active", previousId));
                if (previous) return mayResumeLease ? { ...previous } : { ...previous, attachment: null };
            }
            return {};
        }
        function saveSession(record) {
            requireBackend();
            const id = String(record.jobId || "draft");
            const recovery = {};
            const view = { ...record, backendId, schema: 1 };
            ["pendingTrainingClassCommits", "pendingReviewDispositionCommits", "pendingAnnotationTransactions", "failedReviewDispositionActions"]
                .forEach((name) => { recovery[name] = record[name] || []; delete view[name]; });
            // Keep recovery synchronous and tab-owned: another browser tab must
            // never replace pending transactions while saving its own view.
            write(localStorage, key("recovery", `${id}:${tabId}`), { ...recovery, backendId, jobId: id, tabId });
            write(localStorage, key("view", id), view);
            return record;
        }
        function confirmed(identity) {
            requireBackend();
            return write(sessionStorage, key("active", tabId), { ...identity, backendId, updatedAt: new Date().toISOString() });
        }
        function migrateLegacy(binding) {
            requireBackend();
            const legacy = safeRead(sessionStorage, LEGACY, {});
            if (!legacy.jobId || legacy.jobId !== binding.jobId) return false;
            if (!legacy.snapshotId || legacy.snapshotId !== binding.snapshotId) return false;
            if (legacy.backendId && legacy.backendId !== backendId) return false;
            saveSession({ ...legacy, backendId });
            confirmed({ jobId: binding.jobId, snapshotId: binding.snapshotId });
            write(sessionStorage, `${PREFIX}legacyMigration`, { backendId, jobId: binding.jobId });
            // Keep the legacy copy until migration is independently usable.
            return true;
        }
        const copiedRecoveryKeys = new Set();
        function record(kind, value, owner = backendId) {
            if (!owner) throw new Error("The backend identity has not been verified.");
            if (value == null) write(sessionStorage, key(kind, tabId, owner), null);
            else write(sessionStorage, key(kind, tabId, owner), { ...value, backendId: owner });
        }
        function recoveryKey(kind, id = "") {
            if (!backendId) return "";
            const ownKey = key(kind, `${id}:${tabId}`);
            if (predecessorId && !copiedRecoveryKeys.has(ownKey)) {
                copiedRecoveryKeys.add(ownKey);
                for (const storage of [localStorage, sessionStorage]) {
                    if (storage.getItem(ownKey) !== null) continue;
                    for (const previousId of predecessorIds) {
                        const previousKey = key(kind, `${id}:${previousId}`);
                        if (storage.getItem(previousKey) === null) continue;
                        const previous = safeRead(storage, previousKey);
                        const copy = previous ? { ...previous } : null;
                        if (copy && kind === "annotationSource" && !mayResumeLease) delete copy.lockSessionId;
                        write(storage, ownKey, copy);
                        break;
                    }
                }
            }
            return ownKey;
        }
        async function inventory() {
            let estimate = null;
            let estimateError = "";
            try {
                if (root.navigator?.storage?.estimate) estimate = await root.navigator.storage.estimate();
                else estimateError = "Storage estimates are unavailable in this browser or connection.";
            } catch (error) { estimateError = String(error.message || error); }
            const records = [];
            let metadataBytes = 0;
            for (const [location, storage] of [["local", localStorage], ["tab", sessionStorage]]) {
                try {
                    for (let index = 0; index < storage.length; index += 1) {
                        const name = storage.key(index);
                        if (!name?.startsWith("tator.")) continue;
                        const value = storage.getItem(name) || "";
                        const bytes = 2 * (name.length + value.length);
                        metadataBytes += bytes;
                        const removable = name.startsWith(PREFIX) && name.includes(".view.");
                        records.push({ key: name, location, bytes, removable, value });
                    }
                } catch (_) { /* Denied storage is reported independently of backend inventory. */ }
            }
            return { origin: root.location.origin, estimate, estimateError, metadataBytes, records };
        }
        function forgetView(record) {
            if (record?.location !== "local" || !record.key?.startsWith(PREFIX) || !record.key.includes(".view.")) {
                throw new Error("Pending recovery and source permissions cannot be cleared here.");
            }
            if (localStorage.getItem(record.key) !== record.value) throw new Error("This browser record changed. Refresh before clearing it.");
            localStorage.removeItem(record.key);
        }
        return {
            setBackend(id) { backendId = String(id || ""); },
            getBackend: () => backendId,
            legacyBound: () => safeRead(sessionStorage, `${PREFIX}legacyMigration`, {}).backendId === backendId,
            getTabId: () => tabId,
            sanitizeLegacySource(record) {
                const copy = { ...record };
                if (!mayResumeLease) delete copy.lockSessionId;
                return copy;
            },
            readSession, saveSession, confirmed, migrateLegacy, recoveryKey, inventory, forgetView,
            readActive,
            legacy: () => safeRead(sessionStorage, LEGACY, {}),
            record,
            readOperation: (kind) => {
                if (!backendId) return null;
                if (sessionStorage.getItem(key(kind, tabId)) !== null) return safeRead(sessionStorage, key(kind, tabId));
                if (mayResumeLease) {
                    for (const previousId of predecessorIds) {
                        const previousKey = key(kind, previousId);
                        if (sessionStorage.getItem(previousKey) !== null) return safeRead(sessionStorage, previousKey);
                    }
                }
                return null;
            },
        };
    }
    root.TatorDqeSessionStore = Object.freeze({ create });
}(typeof globalThis !== "undefined" ? globalThis : this));
