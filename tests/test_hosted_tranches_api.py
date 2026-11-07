"""One real HTTP/auth/receipt flow covering the new tranche API boundary."""
from test_hosted_api import application, login, provision, dataset, upload, op, acquire, context


def test_tranche_create_activate_scoped_canvas_submit_and_personal_status(application):
    admin = login(application); user = provision(admin, 'alice'); provision(admin, 'bob')
    alice, bob = login(application, 'alice'), login(application, 'bob')
    ds = dataset(admin); image = upload(admin, ds['id'])
    base = '/api/hosted'
    command = {'operation_id': op(), 'name': 'North block', 'instructions': 'Check all vehicles', 'annotator_id': user['id'],
               'image_ids': [image['id']], 'expected': {image['id']: {'expected_version': image['current_version'], 'generation': image['generation']}}}
    response = admin.post(base + '/datasets/' + ds['id'] + '/tranches', json=command)
    assert response.status_code == 200, response.text
    created = response.json(); tranche = created['tranche']; tid = tranche['id']
    assert admin.post(base + '/datasets/' + ds['id'] + '/tranches', json=command).json() == created
    assert admin.get(base + '/receipts/' + command['operation_id']).json()['result'] == created
    assert alice.get(base + '/tranches/' + tid).status_code == 404
    assert admin.get('/annotate', params={'image': image['id'], 'tranche': tid, 'queue': 'draft'}).status_code == 200
    stale = admin.get('/annotate', params={'image': image['id'], 'queue': 'confirmed'})
    assert stale.status_code == 200 and '"scope_left": true' in stale.text
    assert admin.get(base + '/images/' + image['id'] + '/scope', params={'queue': 'confirmed'}).status_code == 409
    assert bob.get('/annotate', params={'image': image['id'], 'tranche': tid, 'queue': 'draft'}).status_code == 404
    activation = {'operation_id': op(), 'expected_revision': tranche['revision'], 'image_ids': command['image_ids'], 'expected': command['expected'], 'annotator_id': user['id']}
    response = admin.post(base + '/tranches/' + tid + '/activate', json=activation)
    assert response.status_code == 200, response.text
    image = response.json()['images'][0]
    assert bob.get(base + '/tranches/' + tid).status_code == 404
    assert alice.get(base + '/tranches/' + tid).json()['counts']['to_finish'] == 1
    assert alice.get(base + '/queue', params={'dataset_id': ds['id'], 'tranche_id': tid, 'queue': 'work'}).json()['images'][0]['id'] == image['id']
    assert alice.post(base + '/tranches/' + tid + '/close', json={'operation_id': op(), 'expected_revision': 1}).status_code == 403
    unknown = admin.post(base + '/tranches/' + tid + '/edit', json={'operation_id': op(), 'expected_revision': 1, 'name': 'Unsafe', 'instructions': '', 'unexpected': 1})
    assert unknown.status_code == 422
    current = acquire(alice, image)
    saved = alice.post(base + '/images/' + image['id'] + '/save', json={**context(current), 'annotations': {'objects': []}, 'negative': True})
    assert saved.status_code == 200, saved.text
    sent = alice.post(base + '/images/' + image['id'] + '/submit', json=context(saved.json()))
    assert sent.status_code == 200, sent.text
    submitted = sent.json()
    assert alice.get(base + '/images/' + image['id']).status_code == 404
    assert alice.get(base + '/tranches/' + tid).json()['counts']['awaiting_review'] == 1
    member = alice.get(base + '/tranches/' + tid + '/members').json()['members'][0]
    assert member['image'] is None and not member['can_view']
    admin_review = acquire(admin, submitted)
    approved = admin.post(base + '/images/' + image['id'] + '/approve', json=context(admin_review))
    assert approved.status_code == 200, approved.text
    assert alice.get(base + '/tranches').json()['tranches'][0]['counts']['approved'] == 1
    assert alice.get(base + '/datasets/' + ds['id']).status_code == 404
    assert admin.get(base + '/tranches/' + tid + '/members').json()['members'][0]['state'] == 'approved'


def test_recovery_shell_requires_own_matching_handoff_receipt(application):
    admin = login(application); provision(admin, 'alice'); alice = login(application, 'alice')
    ds = dataset(admin); image = upload(admin, ds['id']); other = upload(admin, ds['id'], color='blue')
    current = acquire(admin, image); command = {**context(current), 'reason': 'Recovery shell test'}
    result = admin.post('/api/hosted/images/' + image['id'] + '/archive', json=command)
    assert result.status_code == 200, result.text
    url = '/annotate'
    assert admin.get(url, params={'image': image['id']}).status_code == 404
    assert admin.get(url, params={'image': image['id'], 'recovery': 'unknown-operation'}).status_code == 404
    assert alice.get(url, params={'image': image['id'], 'recovery': command['operation_id']}).status_code == 404
    response = admin.get(url, params={'image': image['id'], 'queue': 'unallocated', 'recovery': command['operation_id']})
    assert response.status_code == 200 and '"scope": null' in response.text
    assert image['filename'] not in response.text and image['media_hash'] not in response.text
    # A receipt for a different archived image cannot unlock the shell.
    current = acquire(admin, other)
    assert admin.post('/api/hosted/images/' + other['id'] + '/archive', json={**context(current), 'reason': 'Second image'}).status_code == 200
    assert admin.get(url, params={'image': other['id'], 'recovery': command['operation_id']}).status_code == 404
