"""Bounded, immutable completed-analysis views; files and ledgers stay authority."""
from __future__ import annotations
import copy
import json
import threading
from collections import OrderedDict
from pathlib import Path
from fastapi import HTTPException
from services.dataset_pruning_cache import file_state


def _readonly(*args, **kwargs):
    raise TypeError('Completed pruning snapshots are read-only')


class FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly
    def __deepcopy__(self, memo): return thaw(self)


class FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _readonly
    def __deepcopy__(self, memo): return thaw(self)


def freeze(value):
    if isinstance(value, (FrozenDict, FrozenList)): return value
    if isinstance(value, dict): return FrozenDict((key, freeze(item)) for key, item in value.items())
    if isinstance(value, list): return FrozenList(freeze(item) for item in value)
    return value


def thaw(value):
    if isinstance(value, dict): return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, list): return [thaw(item) for item in value]
    if isinstance(value, tuple): return tuple(thaw(item) for item in value)
    return copy.deepcopy(value)


def states(paths):
    values = []
    for raw in paths:
        path = Path(raw)
        if path.is_symlink(): raise HTTPException(409, 'Completed pruning artifact path changed')
        values.append((str(path), tuple(file_state(path)) if path.exists() else None))
    return tuple(values)


def verify(binding):
    if states(path for path, _ in binding) != binding:
        raise HTTPException(409, 'Completed pruning artifacts changed; preserve this analysis and recompute it')


class Snapshot:
    def __init__(self, binding, result, source, images):
        self.binding = binding
        self.result, self.source, self.images = freeze(result), freeze(source), freeze(images)
        self.indices = {image['image_id']: i for i, image in enumerate(self.images)}
        if len(self.indices) != len(self.images): raise HTTPException(409, 'Completed pruning image identities are ambiguous')


class CompletedSnapshots:
    """One parsed job, with single-flight admission and no status/other lock calls.

    The384MiB admission limit counts serialized files, not Python heap/RSS.
    The index contains ordinals into the one immutable tree, not copied rows.
    """
    def __init__(self, max_serialized_bytes=384 * 1024**2):
        self.lock = threading.RLock()
        self.max_serialized_bytes = max_serialized_bytes
        self.current = None
        self.observed = {}
        self.counts = {'loads': 0, 'hits': 0, 'drift_rejections': 0}

    def load(self, directory):
        directory = Path(directory)
        # Caller checks status before acquiring this private lock. Never call
        # back into PruningService (service.lock) from here.
        with self.lock:
            identifier = str(directory)
            old = self.observed.get(identifier)
            if old is not None:
                try: verify(old)
                except HTTPException:
                    self.current = None
                    self.counts['drift_rejections'] += 1
                    raise
            if self.current is not None and self.current[0] == identifier:
                verify(self.current[1].binding)
                self.counts['hits'] += 1
                return self.current[1]
            paths = [directory / name for name in ('source.json', 'result.json', 'images.json')]
            initial = states(paths)
            source = json.loads(paths[0].read_text())
            annotation_path = source.get('annotation_states_path') if source.get('annotation_states') is None else None
            if annotation_path: paths.append(Path(annotation_path))
            before = states(paths)
            if before[:3] != initial: raise HTTPException(409, 'Completed pruning artifacts changed during reading')
            result = json.loads(paths[1].read_text())
            images = json.loads(paths[2].read_text())
            if annotation_path: source['annotation_states'] = json.loads(Path(annotation_path).read_text())
            snapshot = Snapshot(before, result, source, images)
            verify(before)
            self.observed[identifier] = before
            self.counts['loads'] += 1
            admitted = sum(state[2] for _, state in before if state is not None) <= self.max_serialized_bytes
            self.current = (identifier, snapshot) if admitted else None
            return snapshot


class BoundedViews:
    """Small detached derived JSON views, never descriptor matrices."""
    def __init__(self, max_entries, max_bytes=64 * 1024**2):
        self.entries = OrderedDict()
        self.max_entries, self.max_bytes, self.bytes = max_entries, max_bytes, 0
        self.counts = {'hits': 0, 'misses': 0}

    def get(self, key):
        field = self.entries.get(key)
        if field is None:
            self.counts['misses'] += 1
            return None
        self.entries.move_to_end(key)
        self.counts['hits'] += 1
        return thaw(field[0])

    def put(self, key, value):
        size = len(json.dumps(value, separators=(',', ':'), allow_nan=False))
        if size > self.max_bytes: return
        old = self.entries.pop(key, None)
        if old: self.bytes -= old[1]
        self.entries[key] = freeze(value), size
        self.bytes += size
        while len(self.entries) > self.max_entries or self.bytes > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old[1]


class ServingContext(tuple):
    def __new__(cls, values, binding):
        obj = super().__new__(cls, values)
        obj.binding = binding
        return obj
