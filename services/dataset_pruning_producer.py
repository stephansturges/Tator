"""Small producer manifests for pruning; never deserialize a full DQE result."""
from pathlib import Path
import json
import hashlib


def publish_attested_component(cache, recipe, rows, artifacts, receipt_path):
    """Bridge completed shared producer artifacts into the same raw cache.

    Internal orchestration supplies an independently qualified terminal receipt;
    this is not an upload endpoint or a shape-based compatibility inference.
    Rows carry actual encoded image SHA and exact current geometry key, so class
    edits and source ordering never make an unrelated vector reusable.
    """
    from services.dataset_pruning import digest
    from services.dataset_pruning_cache import file_state
    receipt_path = Path(receipt_path)
    before = file_state(receipt_path)
    if receipt_path.is_symlink() or before[2] > 8 * 1024 * 1024:
        raise ValueError('Invalid producer receipt')
    payload = receipt_path.read_bytes(); receipt = json.loads(payload)
    if receipt.get('schema') != 'pruning-component-reference-attestation-v1' or receipt.get('status') != 'completed':
        raise ValueError('Only completed, qualified producer components can be published')
    if receipt.get('raw_recipe_digest') != digest(recipe) or receipt.get('artifacts') != artifacts:
        raise ValueError('Producer receipt does not bind this exact raw component recipe/artifact population')
    evidence = receipt.get('producer_equivalence_evidence')
    if not receipt.get('source_row_inventory_sha256') or not isinstance(evidence, list) or not 1 <= len(evidence) <= 32:
        raise ValueError('Producer row content and actual preprocessing/runtime equivalence evidence are required')
    evidence_states = []
    for item in evidence:
        path = Path(item['path']); state = file_state(path)
        if path.is_symlink(): raise ValueError('Producer qualification evidence cannot be a symlink')
        sha = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''): sha.update(chunk)
        if sha.hexdigest() != item['sha256'] or file_state(path) != state:
            raise ValueError('Producer equivalence evidence changed')
        evidence_states.append((path, state))
    row_sha = hashlib.sha256(); count = 0
    def entries():
        nonlocal count
        for row in rows:
            sha = row['image_sha256']; geometry = row.get('geometry_key', '')
            if not isinstance(sha, str) or len(sha) != 64:
                raise ValueError('Producer rows require actual encoded image SHA256')
            raw_recipe = recipe
            if recipe.get('scope') == 'bbox_crop':
                if not geometry: raise ValueError('Object producer rows need their exact geometry key')
            else:
                dimensions = row.get('source_dimensions')
                if not isinstance(dimensions, list) or len(dimensions) != 2 or any(type(v) is not int or v <= 0 for v in dimensions):
                    raise ValueError('Whole-image producer rows need their original dimensions')
                from services.dataset_pruning_adapter import whole_source_recipe
                raw_recipe = whole_source_recipe(recipe, *dimensions)
            canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            row_sha.update(canonical + b'\n'); count += 1
            yield cache.key(sha, raw_recipe, geometry), row['artifact_index'], row['artifact_row']
        if row_sha.hexdigest() != receipt['source_row_inventory_sha256'] or count != receipt['row_count']:
            raise ValueError('Producer row inventory differs from the qualified terminal receipt')
        if file_state(receipt_path) != before or receipt_path.read_bytes() != payload:
            raise ValueError('Producer receipt changed during publication')
        if any(file_state(path) != state for path, state in evidence_states):
            raise ValueError('Producer equivalence evidence changed during publication')
    return cache.put_attested_references(entries(), artifacts, producer_receipt_sha256=hashlib.sha256(payload).hexdigest(),
                                        expected_rows=receipt['row_count'])


def read_summary(path: Path, *, max_bytes=8 * 1024 * 1024):
    """Read the existing first top-level summary with a strict byte bound.

    Current DQE publications put summary first. Unknown layouts fail closed and
    require a small producer manifest; scanning every object is not a fallback.
    """
    decoder = json.JSONDecoder()
    prefix = b""
    with Path(path).open("rb") as handle:
        while len(prefix) < max_bytes:
            chunk = handle.read(min(64 * 1024, max_bytes - len(prefix)))
            if not chunk:
                break
            prefix += chunk
            try:
                text = prefix.decode("utf-8")
            except UnicodeDecodeError:
                continue
            start = text.find('"summary"')
            if start < 0:
                continue
            if text[:start].strip() != "{":
                raise ValueError("DQE result needs a bounded producer summary")
            colon = text.find(":", start + len('"summary"'))
            if colon < 0:
                continue
            value_start = colon + 1
            while value_start < len(text) and text[value_start].isspace():
                value_start += 1
            try:
                value, _ = decoder.raw_decode(text, value_start)
                if not isinstance(value, dict):
                    raise ValueError("Invalid DQE producer summary")
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("DQE producer summary unavailable within bounded read")
