"""Paid image-caption transport, fenced by image ownership and durable reservations.

No SDK retries: once a call is dispatched, an unknown network outcome is never
resent automatically. Pricing is operator-configured and contains no credentials.
"""
from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import re

import httpx

from .artifacts import ArtifactRef
from .errors import DomainError


class PaidCaptionGateway:
    def __init__(self, accounts, artifacts, *, pricing=None, transport=None):
        self.accounts, self.store, self.artifacts = accounts, accounts.store, artifacts
        if pricing is None:
            path = os.environ.get('TATOR_HOSTED_PROVIDER_PRICING')
            pricing = json.loads(Path(path).read_text()) if path else {}
        self.pricing = pricing
        self.transport = transport or self._send
        with self.store.transaction() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS hosted_provider_results (
                call_id TEXT PRIMARY KEY REFERENCES hosted_provider_calls(id),
                result_json TEXT NOT NULL, created_at REAL NOT NULL)''')

    @staticmethod
    def _send(secret, payload):
        # Fixed endpoint, no redirects/proxies, no provider tools or user URLs.
        with httpx.Client(timeout=180, trust_env=False, follow_redirects=False) as client:
            with client.stream('POST', 'https://api.openai.com/v1/responses',
                               headers={'Authorization': 'Bearer ' + secret}, json=payload) as response:
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 4 * 1024 * 1024:
                        raise DomainError('provider_response_too_large', 502)
                    chunks.append(chunk)
                if response.status_code != 200:
                    # Provider error bodies can contain prompts; never relay them.
                    raise DomainError('paid_provider_request_failed', 502)
                return json.loads(b''.join(chunks))

    def catalog(self, actor, dataset_id):
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            self.store._dataset(con, dataset_id)
            if actor.role != 'admin' and not con.execute(
                    "SELECT 1 FROM hosted_images WHERE dataset_id=? AND assignee_id=? AND state IN ('assigned','returned') LIMIT 1",
                    (dataset_id, actor.user_id)).fetchone():
                raise DomainError('dataset_not_found', 404)
            row = con.execute("SELECT * FROM hosted_providers WHERE name='openai'").fetchone()
            if not row or not row['enabled'] or dataset_id not in json.loads(row['allowed_datasets_json']):
                return {'models': []}
            return {'models': [{'id': model, 'max_output_tokens': self.pricing[model]['max_output_tokens'],
                                'image_details': self.pricing[model].get('image_details', ['low', 'high', 'auto'])}
                               for model in json.loads(row['models_json']) if model in self.pricing]}

    def caption(self, actor, context, *, model, prompt, max_output_tokens=2048, detail='high', dispatch_guard=None):
        if not isinstance(prompt, str) or not 1 <= len(prompt.encode()) <= 64000 or detail not in {'low', 'high'}:
            raise DomainError('paid_caption_request_invalid', 422)
        price = self.pricing.get(model)
        required = ('input_microdollars_per_million', 'cached_input_microdollars_per_million',
                    'output_microdollars_per_million', 'input_token_ceiling', 'max_output_tokens')
        if not isinstance(price, dict) or any(type(price.get(key)) is not int or price[key] <= 0 for key in required):
            raise DomainError('provider_pricing_not_configured', 503)
        if type(max_output_tokens) is not int or not 16 <= max_output_tokens <= price['max_output_tokens']:
            raise DomainError('paid_caption_output_limit', 422)
        # Reserve the configured maximum input context rather than a guessed
        # image-token estimate. An operator must set current model rates/limits.
        reservation = (price['input_token_ceiling'] * price['input_microdollars_per_million'] +
                       max_output_tokens * price['output_microdollars_per_million'] + 999_999) // 1_000_000
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            image = dict(self.store._guard(con, actor, context, renew=False))
            media = con.execute('SELECT * FROM hosted_media WHERE sha256=?', (image['media_hash'],)).fetchone()
            reference = ArtifactRef(image['media_hash'], media['size_bytes'])
        identity = {'model': model, 'prompt': prompt, 'max_output_tokens': max_output_tokens,
                    'detail': detail, 'media_hash': reference.sha256, 'pricing': price}
        legacy_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with self.store.read_connection() as con:
            previous = con.execute('SELECT * FROM hosted_provider_calls WHERE actor_id=? AND operation_id=?',
                                   (actor.user_id, context.operation_id)).fetchone()
            prepared = (con.execute("SELECT data_json FROM hosted_provider_audit WHERE kind='media_prepared' AND json_extract(data_json,'$.call_id')=? ORDER BY id LIMIT 1", (previous['id'],)).fetchone()
                        if previous is not None else None)
        from .media import HostedMedia, inspect_media, validate_working_view
        media_views = HostedMedia(self.store, self.artifacts)
        legacy = previous is not None and previous['request_hash'] == legacy_fingerprint and prepared is None
        if legacy:
            # Exact old receipt identity is checked before the new pixel policy.
            fingerprint, view = legacy_fingerprint, None
        elif prepared is not None:
            view = json.loads(prepared[0])['input_identity']
            identity['working_media'] = view
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        elif previous is not None:
            raise DomainError('operation_id_reused', 409)
        else:
            with self.artifacts.open(reference) as stream:
                info = inspect_media(stream)
            if info.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise DomainError('paid_caption_image_format_unsupported', 422)
            view = media_views.resolve(reference)
            identity['working_media'] = view
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        call = self.accounts.reserve(actor, context, provider='openai', model=model,
                                     maximum_microdollars=reservation, request_fingerprint=fingerprint,
                                     input_identity=view)
        with self.store.read_connection() as con:
            prior = con.execute('SELECT result_json FROM hosted_provider_results WHERE call_id=?', (call['id'],)).fetchone()
        if prior:
            return self._current_result(actor, context, json.loads(prior[0]))
        if legacy:
            if self.accounts.release_reserved(call['id'], 'media_policy_upgrade_required'):
                raise DomainError('media_policy_upgrade_required', 409, 'Start a new operation; this legacy call was never dispatched.')
            with self.store.read_connection() as con:
                current = con.execute('SELECT state FROM hosted_provider_calls WHERE id=?', (call['id'],)).fetchone()
                released = con.execute("SELECT 1 FROM hosted_provider_audit WHERE kind='not_dispatched' AND json_extract(data_json,'$.call_id')=? AND json_extract(data_json,'$.reason')='media_policy_upgrade_required'", (call['id'],)).fetchone()
            if current[0] == 'reconciled' and released:
                raise DomainError('media_policy_upgrade_required', 409, 'Start a new operation; this legacy call was never dispatched.')
            raise DomainError('paid_call_outcome_unresolved', 409)
        if call['state'] != 'reserved':
            raise DomainError('paid_call_outcome_unresolved', 409)
        try:
            with self.artifacts.open(media_views.reference(view)) as stream:
                pixels = stream.read()
            validate_working_view(pixels, view, source_sha256=reference.sha256,
                                  source_size_bytes=reference.size_bytes, require_current_compatible=True)
            mime = view['content_type']
            payload = {'model': model, 'store': False, 'service_tier': 'default',
                       'max_output_tokens': max_output_tokens,
                       'input': [{'role': 'user', 'content': [
                           {'type': 'input_text', 'text': prompt},
                           {'type': 'input_image', 'detail': detail,
                            'image_url': f'data:{mime};base64,' + base64.b64encode(pixels).decode('ascii')}]}]}
            secret = self.accounts.dispatch(actor, context, call['id'], guard=dispatch_guard)
        except Exception as error:
            # All preparation failures are definitely pre-dispatch. A pause
            # alone preserves its exact reservation for a later explicit retry.
            if not isinstance(error, DomainError) or error.code != 'runtime_automation_paused':
                self.accounts.release_reserved(call['id'], error.code if isinstance(error, DomainError) else 'paid_caption_preparation_failed')
            raise
        try:
            response = self.transport(secret, payload)
            usage = response.get('usage') or {}
            input_tokens, output_tokens = usage.get('input_tokens'), usage.get('output_tokens')
            cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
            if any(type(number) is not int or number < 0 for number in (input_tokens, output_tokens, cached)) or cached > input_tokens:
                raise DomainError('paid_provider_usage_missing', 502)
            charge = ((input_tokens - cached) * price['input_microdollars_per_million'] +
                      cached * price['cached_input_microdollars_per_million'] +
                      output_tokens * price['output_microdollars_per_million'] + 999_999) // 1_000_000
            caption = '\n'.join(part['text'] for item in response.get('output', []) if item.get('type') == 'message'
                                for part in item.get('content', []) if part.get('type') == 'output_text' and isinstance(part.get('text'), str))
            result = {'call_id': call['id'], 'caption': caption, 'model': model,
                      'image_id': image['id'], 'annotation_version': context.expected_version,
                      'generation': context.generation, 'usage': usage, 'estimated_microdollars': charge,
                      'provider_response_id': response.get('id', ''), 'status': response.get('status'),
                      'truncated': response.get('status') == 'incomplete'}
            # Billing and retained result commit together, even after logout.
            self.accounts.settle(call['id'], actual_microdollars=charge,
                                 provider_request_id=str(response.get('id', '')), result=result)
        except Exception:
            with self.store.read_connection() as con:
                state = con.execute('SELECT state FROM hosted_provider_calls WHERE id=?', (call['id'],)).fetchone()[0]
            if state == 'dispatched':
                self.accounts.settle(call['id'])
            raise DomainError('paid_call_failed_check_billing', 502) from None
        finally:
            secret = None
        return self._current_result(actor, context, result)

    def _current_result(self, actor, context, result):
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            self.accounts.guard_authority(con, actor, context)
        return result

    def stage(self, actor, authority, *, stage_id, model, user_prompt, system_prompt='', image=None,
              source_media_hash, max_output_tokens=2048, detail='high', reasoning_effort=None,
              service_tier='default', dispatch_guard=None):
        """Trusted native caption stage: text-only, full image or derived window.

        The controller supplies a stable stage ID and pixels derived from its
        pinned source. There is deliberately no browser endpoint for this API.
        """
        if (not isinstance(stage_id, str) or not re.fullmatch(r'[A-Za-z0-9._:-]{8,128}', stage_id)
                or not isinstance(user_prompt, str) or not user_prompt.strip()
                or not isinstance(system_prompt, str) or len((user_prompt + system_prompt).encode()) > 128000
                or detail not in {'low', 'high', 'auto', 'original'}
                or reasoning_effort not in {None, 'none', 'minimal', 'low', 'medium', 'high', 'xhigh'}):
            raise DomainError('paid_stage_request_invalid', 422)
        service_tier = 'default' if service_tier == 'standard' else service_tier
        if service_tier != 'default':
            raise DomainError('paid_stage_tier_not_priced', 422)
        price = self.pricing.get(model)
        required = ('input_microdollars_per_million', 'cached_input_microdollars_per_million',
                    'output_microdollars_per_million', 'input_token_ceiling', 'max_output_tokens')
        if not isinstance(price, dict) or any(type(price.get(key)) is not int or price[key] <= 0 for key in required):
            raise DomainError('provider_pricing_not_configured', 503)
        if detail not in price.get('image_details', ['low', 'high', 'auto']):
            raise DomainError('paid_caption_image_detail_not_allowed', 422)
        if type(max_output_tokens) is not int or not 16 <= max_output_tokens <= price['max_output_tokens']:
            raise DomainError('paid_caption_output_limit', 422)
        context = replace(authority, operation_id=stage_id)
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            source = self.accounts.guard_authority(con, actor, context)
            if source['media_hash'] != source_media_hash:
                raise DomainError('paid_stage_source_changed', 409)
        pixels, derived_hash = None, None
        if image is not None:
            from PIL import Image
            if not isinstance(image, Image.Image) or image.width * image.height > 100_000_000:
                raise DomainError('paid_stage_image_invalid', 422)
            buffer = io.BytesIO(); image.convert('RGB').save(buffer, format='PNG'); pixels = buffer.getvalue()
            if len(pixels) > 40 * 1024 * 1024:
                raise DomainError('paid_stage_image_too_large', 422)
            derived_hash = hashlib.sha256(pixels).hexdigest()
        identity = {'model': model, 'user_prompt': user_prompt, 'system_prompt': system_prompt,
                    'source_media_hash': source_media_hash, 'derived_sha256': derived_hash,
                    'max_output_tokens': max_output_tokens, 'detail': detail,
                    'reasoning_effort': reasoning_effort, 'service_tier': service_tier, 'pricing': price}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        reservation = (price['input_token_ceiling'] * price['input_microdollars_per_million'] +
                       max_output_tokens * price['output_microdollars_per_million'] + 999999) // 1000000
        call = self.accounts.reserve(actor, context, provider='openai', model=model,
                                     maximum_microdollars=reservation, request_fingerprint=fingerprint,
                                     replay_settled_stage=True)
        with self.store.read_connection() as con:
            prior = con.execute('SELECT result_json FROM hosted_provider_results WHERE call_id=?', (call['id'],)).fetchone()
        if prior:
            return self._current_result(actor, context, json.loads(prior[0]))
        if call['state'] != 'reserved':
            raise DomainError('paid_call_outcome_unresolved', 409)
        # Exact model inputs remain inspectable and participate in recovery/GC.
        with self.artifacts.exclusive():
            refs = []
            if pixels is not None:
                refs.append(self.artifacts.put_bytes(pixels))
            refs.append(self.artifacts.put_bytes(json.dumps(identity, sort_keys=True).encode()))
            with self.store.transaction() as con:
                self.store._actor(con, actor); self.accounts.guard_authority(con, actor, context)
                for ref in refs:
                    con.execute('INSERT OR IGNORE INTO hosted_artifact_references VALUES (?,?,?,?,?)',
                                ('paid_stage', call['id'], ref.sha256, ref.size_bytes, self.store.clock()))
        content = [{'type': 'input_text', 'text': user_prompt}]
        if pixels is not None:
            content.append({'type': 'input_image', 'detail': detail, 'image_url': 'data:image/png;base64,' + base64.b64encode(pixels).decode('ascii')})
        payload = {'model': model, 'store': False, 'service_tier': 'default', 'max_output_tokens': max_output_tokens,
                   'input': [{'role': 'user', 'content': content}]}
        if system_prompt:
            payload['instructions'] = system_prompt
        if reasoning_effort is not None:
            payload['reasoning'] = {'effort': reasoning_effort}
        try:
            secret = self.accounts.dispatch(actor, context, call['id'], guard=dispatch_guard)
        except DomainError as error:
            if error.code != 'runtime_automation_paused':
                self.accounts.release_reserved(call['id'], error.code)
            raise
        try:
            response = self.transport(secret, payload)
            usage = response.get('usage') or {}
            count_in, count_out = usage.get('input_tokens'), usage.get('output_tokens')
            cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
            if any(type(n) is not int or n < 0 for n in (count_in, count_out, cached)) or cached > count_in:
                raise DomainError('paid_provider_usage_missing', 502)
            charge = ((count_in - cached) * price['input_microdollars_per_million'] + cached * price['cached_input_microdollars_per_million'] + count_out * price['output_microdollars_per_million'] + 999999) // 1000000
            text = '\n'.join(part['text'] for item in response.get('output', []) if item.get('type') == 'message'
                             for part in item.get('content', []) if part.get('type') == 'output_text' and isinstance(part.get('text'), str))
            result = {'call_id': call['id'], 'stage_id': stage_id, 'text': text, 'caption': text,
                      'model': model, 'image_id': context.image_id, 'annotation_version': context.expected_version,
                      'generation': context.generation, 'usage': usage, 'estimated_microdollars': charge,
                      'input_fingerprint': fingerprint, 'input_artifacts': [ref.as_dict() for ref in refs],
                      'provider_response_id': response.get('id', ''), 'status': response.get('status'),
                      'truncated': response.get('status') == 'incomplete', 'output': response.get('output', [])}
            self.accounts.settle(call['id'], actual_microdollars=charge, provider_request_id=str(response.get('id', '')), result=result)
        except Exception:
            with self.store.read_connection() as con:
                state = con.execute('SELECT state FROM hosted_provider_calls WHERE id=?', (call['id'],)).fetchone()[0]
            if state == 'dispatched':
                self.accounts.settle(call['id'])
            raise DomainError('paid_call_failed_check_billing', 502) from None
        finally:
            secret = None
        return self._current_result(actor, context, result)
