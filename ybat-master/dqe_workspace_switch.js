(function (root) {
    "use strict";
    function create(adapters) {
        let active = null;
        let serial = 0;
        async function open(target) {
            if (active) throw new Error("A workspace is already being opened. Wait or cancel it first.");
            const context = {
                ...adapters.context(), target,
                token: ++serial,
                id: root.crypto?.randomUUID?.() || `switch-${Date.now()}-${serial}`,
                controller: new AbortController(),
            };
            active = context;
            let prepared = null;
            let snapshot = null;
            let installed = false;
            let committed = false;
            let ownsBusy = false;
            let heartbeat = null;
            const check = () => {
                if (active !== context || context.controller.signal.aborted) throw new DOMException("Opening cancelled; the previous workspace was kept.", "AbortError");
                if (context.reservationError) throw context.reservationError;
                if (adapters.context().backendId !== context.backendId || adapters.context().apiRoot !== context.apiRoot) throw new Error("The backend changed while opening the workspace.");
            };
            const phase = (value, message) => {
                context.phase = value;
                adapters.status(message);
                adapters.journal({ id: context.id, target, phase: value, old: adapters.identity(), reservation: context.reservation || null });
            };
            try {
                adapters.assertCanSwitch();
                ownsBusy = true;
                adapters.busy(true);
                phase("preparing", "Checking the selected workspace …");
                context.reservation = await adapters.reserve(context);
                if (adapters.heartbeat) heartbeat = root.setInterval(() => {
                    adapters.heartbeat(context).catch((error) => { context.reservationError = error; });
                }, 20000);
                check();
                phase("saving", "Saving the current workspace …");
                await adapters.preserve(context);
                check();
                snapshot = adapters.capture();
                phase("attaching", "Preparing the saved graph and exact dataset …");
                prepared = await adapters.prepare(context);
                check();
                phase("publishing", "Opening the prepared workspace …");
                installed = true;
                await adapters.install(prepared, context);
                check();
                await adapters.commit(context);
                committed = true;
                adapters.confirm(prepared, context);
                adapters.journal(null);
                await adapters.releasePrevious(snapshot, context);
                adapters.status(!prepared.source ? "Plot opened; the dataset is not attached for annotation." : prepared.readOnly ? "Opened read-only; another editor holds the annotation lock." : "Workspace opened.");
                return prepared;
            } catch (error) {
                if (!committed) {
                    let rollbackError = null;
                    if (installed && snapshot) {
                        try { await adapters.rollback(snapshot, context); }
                        catch (failure) { rollbackError = failure; }
                    }
                    await adapters.cancel(context, prepared).catch((cancelError) => {
                        adapters.status(`Opening stopped. The backend reservation will expire: ${cancelError.message || cancelError}`);
                    });
                    if (!rollbackError) adapters.journal(null);
                    else throw new Error(`Opening failed: ${error.message || error}. Restoring the previous view needs attention: ${rollbackError.message || rollbackError}. Recovery metadata was kept.`);
                }
                throw error;
            } finally {
                if (heartbeat) root.clearInterval(heartbeat);
                if (active === context) {
                    active = null;
                    if (ownsBusy) adapters.busy(false);
                }
            }
        }
        return {
            open,
            isBusy: () => Boolean(active),
            operation: () => active,
            cancel() {
                if (active && !["publishing"].includes(active.phase)) active.controller.abort();
            },
        };
    }
    root.TatorDqeWorkspaceSwitch = Object.freeze({ create });
}(typeof globalThis !== "undefined" ? globalThis : this));
