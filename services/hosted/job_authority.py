"""Private authority for admitted admin heavy jobs; never an interactive Actor.

Only the tool coordinator mints grants under a current authenticated admin.
Every later use is bound to the frozen request and the account authority epoch.
"""
from __future__ import annotations

import hashlib
import json

from .errors import DomainError


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def invalidate_restored_jobs(con):
    """Restore cannot carry execution authority into a new installation."""
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "hosted_heavy_grants" in tables:
        con.execute("UPDATE hosted_heavy_grants SET state='interrupted',error='installation_restored' WHERE state='active'")
    if "hosted_tool_analysis" in tables and "hosted_analysis_jobs" in tables:
        con.execute("UPDATE hosted_analysis_jobs SET state='interrupted',error_json=? WHERE state='captured' AND id IN (SELECT job_id FROM hosted_tool_analysis WHERE state IN ('pending','publishing'))", (_json('installation_restored'),))
    if "hosted_tool_analysis" in tables:
        con.execute("UPDATE hosted_tool_analysis SET state='interrupted',error='installation_restored' WHERE state IN ('pending','publishing')")


class HeavyJobAuthority:
    def __init__(self, store):
        self.store = store
        with store.transaction() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS hosted_heavy_grants (
                request_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
                instance_id TEXT NOT NULL, dataset_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL, operation TEXT NOT NULL, adapter_identity TEXT NOT NULL,
                context_json TEXT NOT NULL, job_id TEXT, state TEXT NOT NULL, error TEXT)''')
            con.execute("CREATE TRIGGER IF NOT EXISTS hosted_heavy_grant_scope_immutable BEFORE UPDATE OF request_id,actor_id,authority_epoch,instance_id,dataset_id,snapshot_id,snapshot_sha256,operation,adapter_identity,context_json,job_id ON hosted_heavy_grants BEGIN SELECT RAISE(ABORT,'immutable heavy job scope'); END")

    def mint(self, con, actor, request_id, context, adapter_identity, job_id=None):
        self.store._actor(con, actor, admin=True)
        if not actor.session_id or not actor.instance_id:
            raise DomainError("login_required", 401)
        if (context.background_job_id != request_id or context.background_adapter_identity != adapter_identity
                or context.actor_id != actor.user_id or context.actor_role != 'admin'):
            raise DomainError('heavy_job_scope_invalid', 409)
        snapshot = con.execute('SELECT dataset_id,manifest_json FROM hosted_snapshots WHERE id=?', (context.annotation_version,)).fetchone()
        if not snapshot or snapshot['dataset_id'] != context.dataset_id:
            raise DomainError('heavy_job_scope_invalid', 409)
        user = con.execute('SELECT authority_epoch FROM hosted_users WHERE id=?', (actor.user_id,)).fetchone()
        instance = con.execute("SELECT value FROM hosted_meta WHERE key='instance_id'").fetchone()[0]
        if context.instance_id != instance:
            raise DomainError('heavy_job_scope_invalid', 409)
        con.execute("INSERT INTO hosted_heavy_grants VALUES (?,?,?,?,?,?,?,?,?,?,?,'active',NULL)",
            (request_id, actor.user_id, user[0], instance, context.dataset_id, context.annotation_version,
             hashlib.sha256(snapshot['manifest_json'].encode()).hexdigest(), context.operation,
             adapter_identity, _json(context.as_dict()), job_id))

    def row(self, con, request_id):
        row = con.execute('SELECT * FROM hosted_heavy_grants WHERE request_id=?', (request_id,)).fetchone()
        return dict(row) if row else None

    def validate(self, con, request_id, *, context=None, job_id=None, completed=False):
        grant = self.row(con, request_id)
        if not grant or (job_id is not None and grant['job_id'] != job_id):
            raise DomainError('heavy_job_scope_invalid', 409)
        frozen = json.loads(grant['context_json'])
        if frozen.get('background_adapter_identity') != grant['adapter_identity']:
            raise DomainError('heavy_job_scope_invalid', 409)
        if context is not None and _json(context.as_dict()) != grant['context_json']:
            raise DomainError('heavy_job_scope_invalid', 409)
        if grant['state'] == 'complete' and completed:
            return grant
        if grant['state'] != 'active':
            raise DomainError(grant['error'] or 'heavy_job_authority_revoked', 409)
        user = con.execute('SELECT role,disabled,authority_epoch FROM hosted_users WHERE id=?', (grant['actor_id'],)).fetchone()
        instance = con.execute("SELECT value FROM hosted_meta WHERE key='instance_id'").fetchone()
        if (not user or user['disabled'] or user['role'] != 'admin' or user['authority_epoch'] != grant['authority_epoch']
                or not instance or instance[0] != grant['instance_id']):
            raise DomainError('heavy_job_authority_revoked', 409)
        snapshot = con.execute('SELECT dataset_id,manifest_json FROM hosted_snapshots WHERE id=?', (grant['snapshot_id'],)).fetchone()
        if (not snapshot or snapshot['dataset_id'] != grant['dataset_id']
                or hashlib.sha256(snapshot['manifest_json'].encode()).hexdigest() != grant['snapshot_sha256']):
            raise DomainError('heavy_job_scope_invalid', 409)
        if grant['operation'] == 'dqe_review':
            frozen = json.loads(grant['context_json'])
            parent_id = frozen['model_config'].get('parent_analysis', {}).get('job_id')
            parent = con.execute('SELECT j.snapshot_id,j.state,d.revision,s.manifest_json FROM hosted_analysis_jobs j JOIN hosted_snapshots s ON s.id=j.snapshot_id JOIN hosted_datasets d ON d.id=s.dataset_id WHERE j.id=?', (parent_id,)).fetchone()
            if not parent or parent['state'] != 'complete' or parent['snapshot_id'] != grant['snapshot_id'] or parent['revision'] != json.loads(parent['manifest_json'])['revision']:
                raise DomainError('analysis_snapshot_stale', 409)
        if grant['job_id']:
            job = con.execute('SELECT actor_id,snapshot_id,consumer FROM hosted_analysis_jobs WHERE id=?', (grant['job_id'],)).fetchone()
            if not job or tuple(job) != (grant['actor_id'], grant['snapshot_id'], grant['operation']):
                raise DomainError('heavy_job_scope_invalid', 409)
        return grant

    def recover(self, con, request_id):
        """Recover only an exact already committed native publication receipt."""
        binding = con.execute('SELECT * FROM hosted_tool_analysis WHERE request_id=?', (request_id,)).fetchone()
        if not binding:
            return None
        if binding['state'] == 'complete' and binding['publication_json']:
            return json.loads(binding['publication_json'])
        if not binding['output_json']:
            return None
        owner = json.loads(binding['actor_json'])['user_id']
        receipt = con.execute('SELECT * FROM hosted_receipts WHERE actor_id=? AND operation_id=?',
                              (owner, 'heavy-publish-' + request_id)).fetchone()
        payload = {'job_id': binding['job_id'], 'output': json.loads(binding['output_json'])}
        digest = hashlib.sha256(_json({'operation': 'publish_native_analysis', 'payload': payload}).encode()).hexdigest()
        if not receipt or receipt['operation'] != 'publish_native_analysis' or receipt['request_hash'] != digest:
            return None
        publication = json.loads(receipt['response_json'])
        job = con.execute('SELECT actor_id,state,result_sha256,snapshot_id FROM hosted_analysis_jobs WHERE id=?', (binding['job_id'],)).fetchone()
        if (not job or job['actor_id'] != owner or job['state'] != 'complete'
                or publication.get('job_id') != binding['job_id'] or publication.get('snapshot_id') != job['snapshot_id']
                or publication.get('result', {}).get('sha256') != job['result_sha256']):
            return None
        con.execute("UPDATE hosted_tool_analysis SET state='complete',publication_json=?,error=NULL WHERE request_id=?", (_json(publication), request_id))
        con.execute("UPDATE hosted_heavy_grants SET state='complete',error=NULL WHERE request_id=?", (request_id,))
        return publication

    def terminal(self, con, request_id, state, error):
        """A bound coordinator may terminalize its own job, never publish data."""
        if state not in {'failed', 'cancelled', 'interrupted'}:
            raise ValueError('invalid heavy terminal state')
        if self.recover(con, request_id):
            return
        binding = con.execute('SELECT * FROM hosted_tool_analysis WHERE request_id=?', (request_id,)).fetchone()
        if binding:
            if binding['state'] in {'failed', 'cancelled', 'interrupted'}:
                state, error = binding['state'], binding['error'] or error
            owner = json.loads(binding['actor_json'])['user_id']
            context = json.loads(binding['context_json'])
            con.execute("UPDATE hosted_analysis_jobs SET state=?,error_json=?,updated_at=? WHERE id=? AND actor_id=? AND snapshot_id=? AND consumer=? AND state='captured'",
                (state, _json(error), self.store.clock(), binding['job_id'], owner, context['annotation_version'], context['operation']))
            con.execute("UPDATE hosted_tool_analysis SET state=?,error=? WHERE request_id=? AND state!='complete'", (state, error, request_id))
        con.execute("UPDATE hosted_heavy_grants SET state=?,error=? WHERE request_id=? AND state='active'", (state, error, request_id))

    def publish(self, con, request_id, job_id, output, write):
        previous = self.recover(con, request_id)
        if previous:
            return previous
        grant = self.validate(con, request_id, job_id=job_id)
        binding = con.execute('SELECT * FROM hosted_tool_analysis WHERE request_id=?', (request_id,)).fetchone()
        if (not binding or binding['job_id'] != job_id or not binding['output_json']
                or json.loads(binding['output_json']) != output or binding['state'] not in {'pending', 'publishing'}
                or json.loads(binding['context_json']) != json.loads(grant['context_json'])):
            raise DomainError('heavy_job_scope_invalid', 409)
        publication = write(con)
        payload = {'job_id': job_id, 'output': output}
        digest = hashlib.sha256(_json({'operation': 'publish_native_analysis', 'payload': payload}).encode()).hexdigest()
        con.execute('INSERT INTO hosted_receipts VALUES (?,?,?,?,?,?)',
            (grant['actor_id'], 'heavy-publish-' + request_id, 'publish_native_analysis', digest, _json(publication), self.store.clock()))
        con.execute("UPDATE hosted_tool_analysis SET state='complete',publication_json=?,error=NULL WHERE request_id=?", (_json(publication), request_id))
        con.execute("UPDATE hosted_heavy_grants SET state='complete',error=NULL WHERE request_id=?", (request_id,))
        return publication
