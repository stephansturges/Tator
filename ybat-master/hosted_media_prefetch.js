/* Disposable, authorized image bytes for the next few hosted canvas entries. */
(() => {
    "use strict";
    const IMAGE_LIMIT = 32 * 1024 * 1024;
    const TOTAL_LIMIT = 256 * 1024 * 1024;
    const AHEAD = 8;
    const PARALLEL = 3;
    const HASH = /^[a-f0-9]{64}$/i;

    class HostedMediaPrefetch {
        constructor({request}) {
            if (typeof request !== "function") throw new TypeError("Prefetch needs a request callback.");
            this._request = request;
            this._entries = new Map();
            this._queue = [];
            this._paused = true;
            this._loading = 0;
            this._epoch = 0;
        }

        get stats() {
            return Object.freeze({entries: this._entries.size, bytes: this._bytes(), inFlight: this._loading});
        }

        schedule(ids) {
            const next = [...new Set((Array.isArray(ids) ? ids : []).filter(id => typeof id === "string" && id))].slice(0, AHEAD);
            for (const id of this._entries.keys()) if (!next.includes(id)) this._drop(id);
            this._queue = next;
            for (const id of next) {
                if (!this._entries.has(id)) this._entries.set(id, {
                    id, epoch: this._epoch, state: "queued", cancelled: false,
                    controller: new AbortController(), promise: null, response: null, reader: null,
                    chunks: [], bytes: 0, sourceHash: null, blob: null,
                });
            }
            this._paused = false;
            this._pump();
        }

        pause(keepId = null) {
            this._paused = true;
            // Saving, approving or advancing from A to B must not throw away
            // the images after B: transfers already under way finish, and only
            // ones not yet started wait. The next schedule prunes what is no
            // longer ahead.
            for (const [id, entry] of this._entries) if (id !== keepId && entry.state === "queued") this._drop(id);
            this._queue = this._queue.filter(id => this._entries.has(id));
        }

        clear() {
            this._epoch += 1;
            this.pause();
            for (const id of this._entries.keys()) this._drop(id);
        }

        async take(id, sourceHash) {
            const entry = this._entries.get(id);
            if (!entry) return null;
            if (typeof sourceHash !== "string" || !HASH.test(sourceHash) || entry.state === "queued") {
                this._drop(id); return null;
            }
            // Navigation may share an already-started transfer, but this method
            // never starts speculative or foreground work of its own.
            if (entry.promise) await entry.promise;
            if (!this._current(entry)) return null;
            const blob = entry.state === "ready" && entry.sourceHash === sourceHash.toLowerCase() ? entry.blob : null;
            this._drop(id);
            return blob;
        }

        _current(entry) {
            return !entry.cancelled && entry.epoch === this._epoch && this._entries.get(entry.id) === entry;
        }

        _bytes() {
            let bytes = 0;
            for (const entry of this._entries.values()) bytes += entry.bytes;
            return bytes;
        }

        _cancelBody(entry) {
            try {
                const cancellation = entry.reader ? entry.reader.cancel() : entry.response?.body?.cancel();
                if (cancellation) Promise.resolve(cancellation).catch(() => {});
            } catch (_) { /* A completed or already-locked body needs no cleanup. */ }
        }

        _drop(id) {
            const entry = this._entries.get(id);
            if (!entry) return;
            this._entries.delete(id);
            entry.cancelled = true;
            // A completed transfer only needs its cached bytes released.
            // Aborting its controller can report a successful request as failed.
            if (entry.state !== "ready") {
                entry.controller.abort();
                this._cancelBody(entry);
            }
            entry.chunks.length = 0;
            entry.blob = null;
            entry.bytes = 0;
        }

        _pump() {
            // Nearest images first, a few at a time so one large image does
            // not hold up the ones behind it.
            while (!this._paused && this._loading < PARALLEL) {
                const entry = this._queue.map(id => this._entries.get(id)).find(value => value?.state === "queued");
                if (!entry) return;
                entry.state = "loading";
                this._loading += 1;
                entry.promise = Promise.resolve().then(() => this._load(entry)).finally(() => {
                    this._loading -= 1;
                    this._pump();
                });
            }
        }

        async _load(entry) {
            try {
                if (!this._current(entry)) return;
                entry.response = await this._request(entry.id, entry.controller.signal);
                if (!this._current(entry)) { this._cancelBody(entry); return; }
                const response = entry.response;
                const hash = response.headers.get("X-Tator-Source-Media-Hash") || "";
                const sizeHeader = response.headers.get("Content-Length") || "";
                const expected = /^\d+$/.test(sizeHeader) ? Number(sizeHeader) : NaN;
                if (!response.ok || !HASH.test(hash) || !Number.isSafeInteger(expected) || expected <= 0 || expected > IMAGE_LIMIT) {
                    throw new Error("Image is not eligible for speculative transfer.");
                }
                entry.sourceHash = hash.toLowerCase();
                let blob;
                if (response.body && typeof response.body.getReader === "function") {
                    entry.reader = response.body.getReader();
                    while (true) {
                        const {done, value} = await entry.reader.read();
                        if (!this._current(entry)) return;
                        if (done) break;
                        const size = value?.byteLength;
                        if (!Number.isSafeInteger(size) || size < 0 || entry.bytes + size > expected || entry.bytes + size > IMAGE_LIMIT || this._bytes() + size > TOTAL_LIMIT) {
                            throw new Error("Image exceeded the speculative byte limit.");
                        }
                        entry.bytes += size;
                        entry.chunks.push(value);
                    }
                    if (entry.bytes !== expected) throw new Error("Image transfer was incomplete.");
                    blob = new Blob(entry.chunks, {type: response.headers.get("Content-Type") || "application/octet-stream"});
                } else {
                    // Only a trustworthy bounded length allows the non-stream
                    // fallback; verify the returned bytes before retaining them.
                    blob = await response.blob();
                    if (!this._current(entry)) return;
                    if (blob.size !== expected || this._bytes() + blob.size > TOTAL_LIMIT) throw new Error("Image length changed during transfer.");
                    entry.bytes = blob.size;
                }
                if (!this._current(entry)) return;
                entry.blob = blob;
                entry.state = "ready";
            } catch (_) {
                this._cancelBody(entry);
                entry.controller.abort();
                entry.blob = null;
                entry.bytes = 0;
                if (this._current(entry)) entry.state = "failed";
                // Failed entries stay in this schedule, preventing an automatic
                // retry loop. The ordinary foreground loader remains available.
            } finally {
                try { entry.reader?.releaseLock(); } catch (_) { /* Cancel may still be settling. */ }
                entry.reader = null;
                entry.response = null;
                entry.chunks.length = 0;
            }
        }
    }

    HostedMediaPrefetch.AHEAD = AHEAD;
    window.HostedMediaPrefetch = HostedMediaPrefetch;
})();
