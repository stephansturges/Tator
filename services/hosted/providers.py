"""Server-only paid-provider credentials and durable spending reservations.

Amounts are integer millionths of USD. A reservation is a conservative estimate,
not a provider-enforced charge ceiling. Unknown outcomes retain their reservation
until an administrator reconciles the provider's bill. No caller may supply a key
or a cost estimate through an annotator HTTP endpoint.
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import json
import re
import uuid

from .errors import DomainError
from .types import MutationContext


@dataclass(frozen=True)
class SnapshotAuthority:
    """Internal administrator job authority; never accepted by an HTTP schema."""
    snapshot_id: str
    image_id: str
    expected_version: str
    operation_id: str = 'unused'

    @property
    def generation(self):
        return -1

    @property
    def editor_id(self):
        return 'snapshot:' + self.snapshot_id


class ProviderAccounts:
    def __init__(self, store):
        self.store = store
        with store.transaction() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS hosted_providers (
                name TEXT PRIMARY KEY, revision INTEGER NOT NULL, credential TEXT NOT NULL,
                enabled INTEGER NOT NULL, monthly_limit INTEGER NOT NULL, request_limit INTEGER NOT NULL,
                allowed_datasets_json TEXT NOT NULL, models_json TEXT NOT NULL, updated_at REAL NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS hosted_provider_calls (
                id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                provider TEXT NOT NULL REFERENCES hosted_providers(name), provider_revision INTEGER NOT NULL,
                dataset_id TEXT NOT NULL, image_id TEXT NOT NULL, annotation_version TEXT NOT NULL,
                ownership_generation INTEGER NOT NULL, editor_id TEXT NOT NULL,
                model TEXT NOT NULL, request_hash TEXT NOT NULL, period TEXT NOT NULL,
                reserved INTEGER NOT NULL, charged INTEGER, state TEXT NOT NULL,
                created_at REAL NOT NULL, settled_at REAL, provider_request_id TEXT,
                UNIQUE(actor_id,operation_id))""")
            con.execute("""CREATE TABLE IF NOT EXISTS hosted_provider_audit (
                id INTEGER PRIMARY KEY, actor_id TEXT NOT NULL, provider TEXT NOT NULL,
                kind TEXT NOT NULL, data_json TEXT NOT NULL, created_at REAL NOT NULL)""")
            con.execute("CREATE INDEX IF NOT EXISTS hosted_provider_audit_call ON hosted_provider_audit(json_extract(data_json,'$.call_id'),kind)")

    @staticmethod
    def terminal_batch(row, remote=None):
        """Local failure alone never proves that remote spending has finished."""
        if not row or not row['remote_id'] or not row['input_file_id']:
            return False
        try:
            remote = json.loads(row['remote_json'] or '{}') if remote is None else remote
            return (isinstance(remote, dict) and remote.get('id') == row['remote_id']
                    and remote.get('input_file_id') == row['input_file_id']
                    and remote.get('status') in {'completed', 'failed', 'expired', 'cancelled'}
                    and remote.get('endpoint', '/v1/responses') == '/v1/responses'
                    and isinstance(remote.get('metadata', {}), dict)
                    and remote.get('metadata', {}).get('hosted_job_id', row['id']) == row['id'])
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _has_batches(con):
        return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_paid_batch_rows'").fetchone() is not None

    def mark_batch_terminal(self, con, batch, *, reason='remote_terminal_awaiting_usage'):
        """Called in the transaction retaining the verified terminal response."""
        if not self.terminal_batch(batch):
            return
        rows = con.execute("SELECT c.* FROM hosted_provider_calls c JOIN hosted_paid_batch_rows r ON r.call_id=c.id WHERE r.job_id=? AND c.state='dispatched' AND c.actor_id=? AND c.image_id=r.image_id AND c.provider='openai'", (batch['id'], batch['actor_id'])).fetchall()
        remote = json.loads(batch['remote_json'])
        for call in rows:
            con.execute("UPDATE hosted_provider_calls SET state='unknown' WHERE id=? AND state='dispatched'", (call['id'],))
            con.execute("INSERT INTO hosted_provider_audit(actor_id,provider,kind,data_json,created_at) VALUES (?,?,?,?,?)",
                (call['actor_id'], call['provider'], 'batch_billing_unresolved', json.dumps({'call_id':call['id'], 'job_id':batch['id'],
                    'remote_id':batch['remote_id'], 'remote_status':remote['status'], 'reason':reason}), self.store.clock()))

    def normalize_terminal_batches(self):
        """Idempotent, paged upgrade of historical stranded bills; no network."""
        cursor = ''
        while True:
            with self.store.transaction() as con:
                if not self._has_batches(con):
                    return
                rows = con.execute("SELECT * FROM hosted_paid_batches WHERE id>? AND remote_id IS NOT NULL ORDER BY id LIMIT 100", (cursor,)).fetchall()
                for row in rows:
                    self.mark_batch_terminal(con, row, reason='retained_terminal_response')
            if not rows:
                return
            cursor = rows[-1]['id']

    def billing(self, con, call):
        """Live billing metadata, deliberately separate from proposal identity."""
        batch = None
        if self._has_batches(con):
            linked = con.execute("SELECT b.* FROM hosted_paid_batches b JOIN hosted_paid_batch_rows r ON r.job_id=b.id WHERE r.call_id=?", (call['id'],)).fetchall()
            batch = linked[0] if len(linked) == 1 else None
            bound = bool(linked)
        else:
            bound = False
        eligible = call['state'] in {'reserved','unknown'} and (not bound or self.terminal_batch(batch))
        resolved = call['state'] in {'settled','reconciled'}
        return {'state':call['state'], 'charged':call['charged'], 'reserved':call['reserved'], 'resolved':resolved,
            'can_reconcile':eligible,
            'reason': 'batch_terminal_not_verified' if bound and not resolved and not self.terminal_batch(batch)
                else 'batch_usage_unresolved' if bound and not resolved else 'billing_resolved' if resolved else 'provider_usage_unresolved',
            'batch_job_id':batch['id'] if batch else None, 'remote_batch_id':batch['remote_id'] if batch else None}

    @staticmethod
    def _money(value):
        if type(value) is not int or not 0 <= value <= 10**12:
            raise DomainError("invalid_provider_budget", 422)
        return value

    @staticmethod
    def _public(row):
        return {"name": row["name"], "revision": row["revision"], "has_credential": bool(row["credential"]),
                "enabled": bool(row["enabled"]), "monthly_limit": row["monthly_limit"], "request_limit": row["request_limit"],
                "allowed_datasets": json.loads(row["allowed_datasets_json"]), "models": json.loads(row["models_json"])}

    def configure(self, actor, name, *, expected_revision, operation_id, credential=None,
                  enabled=False, monthly_limit=0, request_limit=0, allowed_datasets=(), models=()):
        if name != "openai" or type(enabled) is not bool:
            raise DomainError("invalid_provider_configuration", 422)
        if credential is not None and (not isinstance(credential, str) or not 16 <= len(credential) <= 2048 or any(c.isspace() for c in credential)):
            raise DomainError("invalid_provider_credential", 422)
        monthly_limit, request_limit = self._money(monthly_limit), self._money(request_limit)
        datasets, models = sorted(set(allowed_datasets)), sorted(set(models))
        if len(datasets) > 10000 or len(models) > 100 or any(not isinstance(v, str) or not re.fullmatch(r"[a-zA-Z0-9._:-]{1,160}", v) for v in models):
            raise DomainError("invalid_provider_configuration", 422)
        if request_limit > monthly_limit or (enabled and (not request_limit or not models or not datasets)):
            raise DomainError("invalid_provider_budget", 422)
        payload = dict(name=name, expected_revision=expected_revision, enabled=enabled, monthly_limit=monthly_limit,
                       request_limit=request_limit, allowed_datasets=datasets, models=models,
                       credential_digest=hashlib.sha256(credential.encode()).hexdigest() if credential else None)
        def update(con):
            row = con.execute("SELECT * FROM hosted_providers WHERE name=?", (name,)).fetchone()
            if (row["revision"] if row else 0) != expected_revision:
                raise DomainError("provider_configuration_changed", 409)
            for dataset_id in datasets:
                self.store._dataset(con, dataset_id)
            secret = credential if credential is not None else row["credential"] if row else ""
            if enabled and not secret:
                raise DomainError("provider_credential_required", 422)
            con.execute("INSERT INTO hosted_providers VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET revision=excluded.revision,credential=excluded.credential,enabled=excluded.enabled,monthly_limit=excluded.monthly_limit,request_limit=excluded.request_limit,allowed_datasets_json=excluded.allowed_datasets_json,models_json=excluded.models_json,updated_at=excluded.updated_at",
                        (name, expected_revision + 1, secret, int(enabled), monthly_limit, request_limit, json.dumps(datasets), json.dumps(models), self.store.clock()))
            con.execute("INSERT INTO hosted_provider_audit(actor_id,provider,kind,data_json,created_at) VALUES (?,?,?,?,?)",
                        (actor.user_id, name, "configured", json.dumps({k: v for k, v in payload.items() if k != "credential_digest"}), self.store.clock()))
            return self._public(con.execute("SELECT * FROM hosted_providers WHERE name=?", (name,)).fetchone())
        return self.store._command(actor, operation_id, "configure_provider", payload, update, admin=True)

    def list(self, actor):
        period = datetime.fromtimestamp(self.store.clock(), timezone.utc).strftime("%Y-%m")
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            result = []
            for row in con.execute("SELECT * FROM hosted_providers ORDER BY name"):
                item = self._public(row)
                summary = con.execute("SELECT COALESCE(SUM(CASE WHEN charged IS NULL THEN reserved ELSE charged END),0),SUM(CASE WHEN state IN ('reserved','dispatched','unknown') THEN 1 ELSE 0 END) FROM hosted_provider_calls WHERE provider=? AND period=?", (row["name"], period)).fetchone()
                item.update(period=period, spent_or_reserved=summary[0], unresolved_calls=summary[1] or 0)
                result.append(item)
            return result

    def guard_authority(self, con, actor, context, *, renew=False):
        if isinstance(context, MutationContext):
            return self.store._guard(con, actor, context, renew=renew)
        if not isinstance(context, SnapshotAuthority):
            raise DomainError('provider_authority_invalid', 403)
        self.store._actor(con, actor, admin=True)
        row = con.execute('SELECT manifest_json FROM hosted_snapshots WHERE id=?', (context.snapshot_id,)).fetchone()
        if row is None:
            raise DomainError('snapshot_not_found', 404)
        manifest = json.loads(row[0])
        self.store._dataset(con, manifest['dataset_id'])
        frozen = next((item for item in manifest['images'] if item['id'] == context.image_id), None)
        image = self.store._image(con, context.image_id)
        if (frozen is None or frozen['version_id'] != context.expected_version
                or image['dataset_id'] != manifest['dataset_id'] or image['state'] != 'confirmed'
                or image['current_version'] != context.expected_version or image['media_hash'] != frozen['media_hash']
                or con.execute('SELECT 1 FROM hosted_archived_images WHERE image_id=?', (image['id'],)).fetchone()):
            raise DomainError('paid_snapshot_image_changed', 409)
        return image

    def reserve(self, actor, context, *, provider, model, maximum_microdollars, request_fingerprint,
                replay_settled_stage=False, input_identity=None):
        """Trusted gateway only: fingerprint includes the complete immutable model input."""
        maximum = self._money(maximum_microdollars)
        if not maximum or not re.fullmatch(r"[0-9a-f]{64}", request_fingerprint):
            raise DomainError("provider_estimate_required", 422)
        period = datetime.fromtimestamp(self.store.clock(), timezone.utc).strftime("%Y-%m")
        with self.store.transaction() as con:
            self.store._actor(con, actor)
            image = self.guard_authority(con, actor, context, renew=True)
            prior = con.execute("SELECT * FROM hosted_provider_calls WHERE actor_id=? AND operation_id=?", (actor.user_id, context.operation_id)).fetchone()
            rearm = False
            if prior:
                if (prior["provider"], prior["model"], prior["request_hash"], prior["reserved"], prior["image_id"], prior["annotation_version"]) != (provider, model, request_fingerprint, maximum, context.image_id, context.expected_version):
                    raise DomainError("operation_id_reused", 409)
                same_editor = (prior['ownership_generation'], prior['editor_id']) == (context.generation, context.editor_id)
                rearm = bool(replay_settled_stage and prior['state'] == 'reconciled' and prior['charged'] == 0
                    and con.execute("SELECT 1 FROM hosted_provider_audit WHERE kind='not_dispatched' AND json_extract(data_json,'$.call_id')=? AND id>COALESCE((SELECT MAX(id) FROM hosted_provider_audit WHERE kind='stage_readmitted' AND json_extract(data_json,'$.call_id')=?),0)", (prior['id'], prior['id'])).fetchone())
                # A private caption job may replay a completed immutable stage
                # after its editor reconnects. Current authority was checked above;
                # the historical bill and original editor remain unchanged.
                if not same_editor and not rearm and not (replay_settled_stage and prior['state'] == 'settled'
                        and con.execute('SELECT 1 FROM hosted_provider_results WHERE call_id=?', (prior['id'],)).fetchone()):
                    if replay_settled_stage:
                        raise DomainError('paid_call_outcome_unresolved', 409)
                    raise DomainError('operation_id_reused', 409)
                if not rearm:
                    return {"id": prior["id"], "state": prior["state"]}
            row = con.execute("SELECT * FROM hosted_providers WHERE name=?", (provider,)).fetchone()
            if not row or not row["enabled"] or not row["credential"]:
                raise DomainError("paid_provider_disabled", 409)
            if image["dataset_id"] not in json.loads(row["allowed_datasets_json"]) or model not in json.loads(row["models_json"]):
                raise DomainError("paid_provider_not_allowed", 403)
            allocated = con.execute("SELECT COALESCE(SUM(CASE WHEN charged IS NULL THEN reserved ELSE charged END),0) FROM hosted_provider_calls WHERE provider=? AND period=?", (provider, period)).fetchone()[0]
            if maximum > row["request_limit"] or allocated + maximum > row["monthly_limit"]:
                raise DomainError("paid_provider_budget_exceeded", 409)
            if rearm:
                # Only a provably unsent stage may be admitted again. A billing
                # reconciliation of an unknown/dispatched call never qualifies.
                con.execute("UPDATE hosted_provider_calls SET state='reserved',charged=NULL,settled_at=NULL,provider_revision=?,period=?,ownership_generation=?,editor_id=? WHERE id=?",
                            (row['revision'], period, context.generation, context.editor_id, prior['id']))
                # This newer audit event consumes the unsent proof. A future
                # unknown call reconciled to zero cannot reuse the old proof.
                con.execute("INSERT INTO hosted_provider_audit(actor_id,provider,kind,data_json,created_at) VALUES (?,?,?,?,?)",
                            (actor.user_id, provider, 'stage_readmitted', json.dumps({'call_id': prior['id'], 'generation': context.generation, 'editor_id': context.editor_id}), self.store.clock()))
                return {'id': prior['id'], 'state': 'reserved'}
            call_id = uuid.uuid4().hex
            con.execute("INSERT INTO hosted_provider_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,'reserved',?,NULL,NULL)",
                        (call_id, actor.user_id, context.operation_id, provider, row["revision"], image["dataset_id"], context.image_id,
                         context.expected_version, context.generation, context.editor_id, model, request_fingerprint, period, maximum, self.store.clock()))
            if input_identity is not None:
                con.execute("INSERT INTO hosted_provider_audit(actor_id,provider,kind,data_json,created_at) VALUES (?,?,?,?,?)",
                    (actor.user_id, provider, 'media_prepared', json.dumps({'call_id':call_id,'input_identity':input_identity}, allow_nan=False), self.store.clock()))
            return {"id": call_id, "state": "reserved"}

    def dispatch(self, actor, context, call_id, *, guard=None):
        """One-time credential handoff to a trusted in-process gateway, never an HTTP response."""
        with self.store.transaction() as con:
            self.store._actor(con, actor)
            self.guard_authority(con, actor, context, renew=True)
            call = con.execute("SELECT * FROM hosted_provider_calls WHERE id=? AND actor_id=?", (call_id, actor.user_id)).fetchone()
            if not call or (call["image_id"], call["annotation_version"], call["operation_id"], call["ownership_generation"], call["editor_id"]) != (context.image_id, context.expected_version, context.operation_id, context.generation, context.editor_id):
                raise DomainError("provider_call_not_found", 404)
            if call["state"] != "reserved":
                raise DomainError("provider_call_already_dispatched", 409)
            row = con.execute("SELECT * FROM hosted_providers WHERE name=?", (call["provider"],)).fetchone()
            if not row["enabled"] or row["revision"] != call["provider_revision"]:
                raise DomainError("provider_configuration_changed", 409)
            if guard is not None:
                guard(con)
            con.execute("UPDATE hosted_provider_calls SET state='dispatched' WHERE id=?", (call_id,))
            return row["credential"]

    def release_reserved(self, call_id, reason):
        """Trusted dispatch refusal: no provider call was admitted or billed."""
        with self.store.transaction() as con:
            call = con.execute("SELECT * FROM hosted_provider_calls WHERE id=?", (call_id,)).fetchone()
            if not call or call['state'] != 'reserved':
                return False
            con.execute("UPDATE hosted_provider_calls SET state='reconciled',charged=0,settled_at=? WHERE id=?", (self.store.clock(), call_id))
            con.execute("INSERT INTO hosted_provider_audit(actor_id,provider,kind,data_json,created_at) VALUES (?,?,?,?,?)",
                        (call['actor_id'], call['provider'], 'not_dispatched', json.dumps({'call_id': call_id, 'reason': str(reason)[:200]}), self.store.clock()))
            return True

    def settle(self, call_id, *, actual_microdollars=None, provider_request_id="", result=None):
        """Trusted transport reports usage even after logout; no annotation authority is granted."""
        charged = None if actual_microdollars is None else self._money(actual_microdollars)
        if len(provider_request_id) > 200:
            raise ValueError("Invalid provider request identifier")
        with self.store.transaction() as con:
            call = con.execute("SELECT * FROM hosted_provider_calls WHERE id=?", (call_id,)).fetchone()
            if not call or call["state"] not in {"dispatched", "unknown"}:
                raise DomainError("provider_call_not_dispatched", 409)
            con.execute("UPDATE hosted_provider_calls SET charged=?,state=?,settled_at=?,provider_request_id=? WHERE id=?",
                        (charged, "settled" if charged is not None else "unknown", self.store.clock(), provider_request_id, call_id))
            if result is not None:
                con.execute("INSERT INTO hosted_provider_results VALUES (?,?,?)", (call_id, json.dumps(result, allow_nan=False), self.store.clock()))
            if charged is not None and charged > call["reserved"]:
                # Underestimation is visible and pauses further paid use; it is
                # never hidden by clamping the observed charge to our estimate.
                con.execute("UPDATE hosted_providers SET enabled=0,revision=revision+1 WHERE name=?", (call["provider"],))

    def settle_batch_row(self, job_id, image_id, call_id, *, actual_microdollars, provider_request_id, output):
        """Serialize collection with manual billing; retain output in the same commit."""
        charged = None if actual_microdollars is None else self._money(actual_microdollars)
        if not isinstance(provider_request_id, str) or len(provider_request_id) > 200:
            raise DomainError('paid_batch_request_id_invalid', 502)
        with self.store.transaction() as con:
            batch = con.execute('SELECT * FROM hosted_paid_batches WHERE id=?', (job_id,)).fetchone()
            row = con.execute('SELECT call_id FROM hosted_paid_batch_rows WHERE job_id=? AND image_id=?', (job_id,image_id)).fetchone()
            call = con.execute('SELECT * FROM hosted_provider_calls WHERE id=?', (call_id,)).fetchone()
            if (not self.terminal_batch(batch) or not row or row['call_id'] != call_id or not call
                    or (call['actor_id'],call['image_id'],call['provider']) != (batch['actor_id'],image_id,'openai')):
                raise DomainError('paid_batch_billing_identity_invalid', 409)
            if call['state'] in {'dispatched','unknown'}:
                con.execute("UPDATE hosted_provider_calls SET charged=?,state=?,settled_at=?,provider_request_id=? WHERE id=?",
                    (charged, 'settled' if charged is not None else 'unknown', self.store.clock(), provider_request_id, call_id))
                if charged is not None and charged > call['reserved']:
                    con.execute('UPDATE hosted_providers SET enabled=0,revision=revision+1 WHERE name=?', (call['provider'],))
            elif call['state'] not in {'settled','reconciled'}:
                raise DomainError('provider_call_not_dispatched', 409)
            if call['state'] in {'settled','reconciled'} and charged is not None and charged != call['charged']:
                audit = {'call_id':call_id, 'job_id':job_id, 'remote_id':batch['remote_id'],
                    'confirmed_charge':call['charged'], 'confirmed_state':call['state'], 'reported_charge':charged,
                    'provider_request_id':provider_request_id}
                encoded = json.dumps(audit, sort_keys=True, allow_nan=False)
                if not con.execute("SELECT 1 FROM hosted_provider_audit WHERE kind='batch_usage_after_billing' AND data_json=?", (encoded,)).fetchone():
                    con.execute('INSERT INTO hosted_provider_audit(actor_id,provider,kind,data_json,created_at) VALUES (?,?,?,?,?)',
                        (call['actor_id'],call['provider'],'batch_usage_after_billing',encoded,self.store.clock()))
            con.execute('UPDATE hosted_paid_batch_rows SET output_json=? WHERE job_id=? AND image_id=?',
                (json.dumps(output, sort_keys=True, allow_nan=False),job_id,image_id))

    def calls(self, actor, *, limit=100, offset=0, unresolved_only=False):
        if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 500 or offset < 0:
            raise DomainError("invalid_pagination", 422)
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            rows = con.execute("SELECT id,actor_id,provider,dataset_id,image_id,model,period,reserved,charged,state,created_at,provider_request_id FROM hosted_provider_calls WHERE (?=0 OR state IN ('reserved','dispatched','unknown')) ORDER BY created_at DESC,id LIMIT ? OFFSET ?", (int(unresolved_only),limit,offset)).fetchall()
            return [{**dict(row), **self.billing(con,row)} for row in rows]

    def reconcile(self, actor, call_id, *, charged, note, operation_id):
        charged = self._money(charged)
        if not isinstance(note, str) or not 3 <= len(note.strip()) <= 2000:
            raise DomainError("provider_billing_note_required", 422)
        def apply(con):
            call = con.execute("SELECT * FROM hosted_provider_calls WHERE id=?", (call_id,)).fetchone()
            if not call or call["state"] not in {"reserved", "unknown"}:
                raise DomainError("provider_call_not_unresolved", 409)
            if not self.billing(con,call)['can_reconcile']:
                raise DomainError('paid_batch_terminal_not_verified', 409)
            con.execute("UPDATE hosted_provider_calls SET charged=?,state='reconciled',settled_at=? WHERE id=?", (charged, self.store.clock(), call_id))
            con.execute("INSERT INTO hosted_provider_audit(actor_id,provider,kind,data_json,created_at) VALUES (?,?,?,?,?)",
                        (actor.user_id, call["provider"], "billing_reconciled", json.dumps({"call_id": call_id, "charged": charged, "note": note}), self.store.clock()))
            return {"id": call_id, "state": "reconciled", "charged": charged}
        return self.store._command(actor, operation_id, "reconcile_provider_call", {"id": call_id, "charged": charged, "note": note}, apply, admin=True)

    @staticmethod
    def after_restore(con):
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "hosted_providers" in tables:
            con.execute("UPDATE hosted_providers SET enabled=0,revision=revision+1")
            con.execute("UPDATE hosted_provider_calls SET state='unknown' WHERE state IN ('reserved','dispatched')")
