/* Progressive, bidirectional keyset queue. No annotation or canvas state is owned here. */
(() => {
    "use strict";
    // Workspace context carried through the canvas so Back restores the same
    // view, search and incoming filter. "q" and "annotator" also scope the queue.
    const CONTEXT_KEYS = ["view", "q", "filter", "annotator"];
    const QUEUE_VIEWS = {unallocated: "incoming", incoming: "incoming", confirmed: "confirmed", review: "submitted"};
    function pageContext(search = location.search) {
        const params = new URLSearchParams(search);
        // "tranche" is kept only for the way back; the queue scope owns its tranche.
        return Object.fromEntries([...CONTEXT_KEYS, "tranche"].filter(key => params.get(key)).map(key => [key, params.get(key)]));
    }
    function workspaceUrl(scope, context = {}, at = null) {
        if (!scope?.dataset_id) return "/";
        const params = new URLSearchParams({dataset: scope.dataset_id, view: context.view || (scope.tranche_id ? "tranches" : QUEUE_VIEWS[scope.queue] || "tranches")});
        // A tranche image opened from Incoming returns to Incoming, not to its tranche.
        const tranche = scope.tranche_id && (!context.view || ["tranches", "mine", "submitted"].includes(context.view)) ? scope.tranche_id : context.tranche;
        if (tranche) params.set("tranche", tranche);
        for (const key of ["q", "filter", "annotator"]) if (context[key]) params.set(key, context[key]);
        if (at) params.set("at", at);
        return `/?${params}`;
    }
    class HostedQueue {
        constructor({api, scope, onAppend, onReconcile, context = pageContext(), filterByAnnotator = false}) {
            this.api = api; this.scope = Object.freeze({...scope}); this.onAppend = onAppend; this.onReconcile = onReconcile; this.rows = []; this.next = null; this.previous = null; this.loading = null;
            this.context = Object.freeze({...context});
            // Previous/Next stay inside the search and annotator the user came from.
            this.filters = {...(context.q ? {q: context.q} : {}), ...(filterByAnnotator && context.annotator ? {annotator_id: context.annotator} : {})};
        }
        params(extra = {}) { return new URLSearchParams({...this.scope, ...(this.scope.tranche_id ? {} : {tranche_id: ""}), ...this.filters, limit: "100", ...extra}); }
        async fetch(extra) {
            const page = await this.api(`/api/hosted/queue?${this.params(extra)}`);
            if (!Array.isArray(page.images) || page.scope?.dataset_id !== this.scope.dataset_id || (page.scope.tranche_id || null) !== (this.scope.tranche_id || null) || page.scope.queue !== this.scope.queue) throw new Error("The server returned a different work queue. Return to the workspace and reopen this tranche.");
            return page;
        }
        async open(imageId) {
            let page;
            try { page = await this.fetch({anchor_id: imageId}); }
            catch (error) {
                // An image that does not match the search keeps its unfiltered queue.
                if (error.code !== "image_left_queue" || !Object.keys(this.filters).length) throw error;
                this.filters = {}; page = await this.fetch({anchor_id: imageId});
            }
            this.rows = page.images; this.next = page.next_cursor; this.previous = page.previous_cursor;
            return this.rows;
        }
        async more(direction = "forward") {
            // Serialize appends and prefetch preparation; a stale second click
            // must not append the same cursor page twice or replace the canvas.
            while (this.loading) await this.loading;
            const cursor = direction === "backward" ? this.previous : this.next;
            if (!cursor) return;
            this.loading = (async () => {
                const page = await this.fetch({cursor, direction});
                const seen = new Set(this.rows.map(row => row.id)), added = page.images.filter(row => !seen.has(row.id));
                if (direction === "backward") { this.rows.unshift(...added); this.previous = page.previous_cursor; }
                else { this.rows.push(...added); this.next = page.next_cursor; }
                this.onAppend?.(added, {prepend: direction === "backward"});
            })();
            try { await this.loading; } finally { this.loading = null; }
        }
        async refresh(id, isCurrent = () => true) {
            while (this.loading) await this.loading;
            this.loading = (async () => {
                const [before, after] = await Promise.all([this.fetch({anchor_id: id, direction: "backward"}), this.fetch({anchor_id: id})]);
                if (!isCurrent()) return;
                const seen = new Set(), rows = [...before.images, ...after.images].filter(row => { if (seen.has(row.id)) return false; seen.add(row.id); return true; });
                this.rows = rows; this.previous = before.previous_cursor; this.next = after.next_cursor;
                this.onReconcile?.(rows);
            })();
            try { await this.loading; } finally { this.loading = null; }
        }
        async neighbor(id, delta = 1) {
            let index = this.rows.findIndex(row => row.id === id);
            if (index < 0) throw new Error("This image left the current work queue. Return to the tranche.");
            const direction = delta < 0 ? "backward" : "forward";
            while (index + delta < 0 && this.previous || index + delta >= this.rows.length && this.next) {
                await this.more(direction); index = this.rows.findIndex(row => row.id === id);
            }
            return this.rows[index + delta] || null;
        }
        peek(id, count = 2) { const index = this.rows.findIndex(row => row.id === id); return index < 0 ? [] : this.rows.slice(index + 1, index + count + 1).map(row => row.id); }
        async ahead(id, count = 2) { const index = this.rows.findIndex(row => row.id === id); if (index >= 0 && index + count >= this.rows.length && this.next) await this.more(); }
        remove(id) { this.rows = this.rows.filter(row => row.id !== id); }
        row(id) { return this.rows.find(row => row.id === id) || null; }
        async authorize(id) { return this.api(`/api/hosted/images/${encodeURIComponent(id)}/scope?${new URLSearchParams({queue: this.scope.queue, ...(this.scope.tranche_id ? {tranche_id: this.scope.tranche_id} : {})})}`); }
        url(id) {
            const params = new URLSearchParams({image: id, queue: this.scope.queue, ...(this.scope.tranche_id ? {tranche: this.scope.tranche_id} : {}), dataset: this.scope.dataset_id});
            for (const key of CONTEXT_KEYS) if (this.context[key]) params.set(key, this.context[key]);
            return `/annotate?${params}`;
        }
        back(at = null) { return workspaceUrl(this.scope, this.context, at); }
        get backUrl() { return this.back(); }
    }
    HostedQueue.pageContext = pageContext;
    HostedQueue.workspaceUrl = workspaceUrl;
    window.HostedQueue = HostedQueue;
})();
