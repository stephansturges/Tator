/* Authenticated hosting adapter; the established pruning controls stay intact. */
(async () => {
    "use strict";
    const boot = window.TatorPruningBoot, notice = document.getElementById("hostedPruningNotice");
    let context, instance;
    const base = `/api/hosted/analysis/jobs/${encodeURIComponent(boot.job_id)}`;
    async function scopedFetch(url, options = {}) {
        const target = new URL(url, location.href);
        if (target.origin !== location.origin) throw new Error("Use this Tator server for pruning.");
        const headers = new Headers(options.headers);
        headers.set("x-tator-csrf", boot.csrf); headers.set("x-tator-instance", boot.instance_id);
        return fetch(target, { ...options, headers, credentials: "same-origin" });
    }
    async function json(path) {
        const response = await scopedFetch(path), value = await response.json();
        if (!response.ok) throw new Error(value.detail || value.error || "Could not load the analysis.");
        return value;
    }
    async function verify() {
        const session = await json("/api/hosted/session");
        if (session.user.user_id !== boot.user_id || session.instance_id !== boot.instance_id || session.csrf !== boot.csrf) {
            throw new Error("Your session changed. Reopen pruning from the dataset workspace.");
        }
    }
    try {
        const [job, meta] = await Promise.all([json(base), json(base + "/view")]);
        const dataset = await json(`/api/hosted/datasets/${encodeURIComponent(job.dataset_id)}`);
        context = { apiRoot: location.origin + base, backendId: `${boot.instance_id}:${boot.user_id}`,
            annotationSource: { kind: "dataset", id: dataset.id }, revision: job.request.dataset_revision,
            datasetName: dataset.name, imageCount: job.confirmed_count, inventory: { image_count: job.confirmed_count } };
        if (job.stale) notice.textContent = "This analysis describes an earlier confirmed version. You can inspect it; run a new analysis before changing its removal plan.";
        instance = window.TatorDatasetPruning.create({ context: () => context, fetch: scopedFetch,
            initialJobId: meta.native_job_id, verifyBackend: verify,
            checkpoint: async () => {
                await verify();
                const current = await json(base);
                if (current.stale) throw new Error("Confirmed data changed. Start a new analysis to use its compatible cached features.");
            },
            getRecipe: () => job.request?.settings?.dqe_recipe || {},
            profileContext: () => context.annotationSource,
            openDQE: () => { location.assign("/?analysis=" + encodeURIComponent(boot.job_id)); },
            openIngestionProfiles: () => { location.assign("/?panel=profiles&analysis=" + encodeURIComponent(boot.job_id)); },
        });
        await instance.activate();
    } catch (error) { notice.textContent = error.message; notice.dataset.error = "true"; }
    window.addEventListener("pagehide", () => instance?.destroy(), { once: true });
})();
