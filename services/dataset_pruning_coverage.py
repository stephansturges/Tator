"""Frozen appearance accounting and cache-only, budgeted image selection.

Grouping defines counting units, never indirect witnesses. Every original box
is compared directly with final retainers. The ranking graph is deliberately
frozen and approximate; exact expansion changes binary coverage, not ranking.
No encoder, trainer, residualizer, source writer or model loader is reachable.
"""
from __future__ import annotations

import hashlib
import heapq
import inspect
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import zlib
from collections import defaultdict
from contextlib import ExitStack

import numpy as np

from services.dataset_pruning import PruningConfig, digest

SCHEMA = "pruning-direct-appearance-coverage-v3"
# Accounting arrays and existing full certificates remain valid. Plans made
# before every positive witness was certified must nevertheless be reapplied.
QUALIFICATION = "authoritative-direct-witness-v2"
EXACT_RARITY = "source-wide-exact-nearest-five-exemplars-v1"
RANK_K = 64
BLOCK = 256
FROZEN_ARRAYS = ("object-rank-images", "object-rank-scores", "object-rank-rows",
                 "scene-rank-images", "scene-rank-scores", "group-of", "representatives", "weights", "isolation")


class CoverageCancelled(InterruptedError):
    pass


class CoverageUnavailable(RuntimeError):
    pass


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise CoverageCancelled("Coverage calculation cancelled")


def _notify(progress, done, total, phase):
    if progress is None:
        return
    try:
        inspect.signature(progress).bind(done, total, phase)
    except (TypeError, ValueError):
        progress(done / max(1, total), phase)
    else:
        progress(done, total, phase)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    os.replace(temporary, path)


def _state(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _array(path, shape, dtype, fill=None):
    result = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        result[:] = fill
    return result


def _save_array(path, values):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as target:
        np.save(target, values, allow_pickle=False)
    os.replace(temporary, path)


def _checkpoint(path, binding):
    if path.is_file():
        result = json.loads(path.read_text())
        if result.get("binding") != binding:
            raise ValueError("Coverage checkpoint belongs to another source or feature generation")
        return result
    return {"binding": binding, "classes": {}}


def _midranks(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return values
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end) / (2 * len(values))
        start = end
    return result


def _cosine_screen_error(width):
    """Conservative bound between fp32 and authoritative fp64 cosines.

    Both calculations use the SAME original fp32 values and fp64 norms; no
    normalized fp32 descriptors are stored. gamma_(2D) bounds a multiply/add
    dot product using Cauchy-Schwarz for its absolute-product sum. The fp64
    term covers its dot, norm and division rounding. Nonfinite screens never
    reject candidates. Final comparisons, means and ties remain fp64.
    """
    single = 2 * width * (np.finfo(np.float32).eps / 2)
    double = 2 * width * (np.finfo(np.float64).eps / 2)
    if single >= 1 or double >= 1:
        return math.inf
    g32, g64 = single / (1 - single), double / (1 - double)
    return float(np.nextafter(g32 * (1 + 16 * g64) + 16 * g64, math.inf))


class CoverageIndex:
    def __init__(self, directory, images, scene, objects, config, binding, cancelled=None):
        self.directory = Path(directory)
        self.images = [{**im, "protection_errors": list(im.get("protection_errors", [])),
                        "objects": [{**obj, "box": list(obj["box"])} for obj in im["objects"]]} for im in images]
        images = self.images
        self.scene = scene
        self.objects = objects
        self.config = config
        self.binding = binding
        self.image_ids = [str(image["image_id"]) for image in images]
        self.id_index = {identifier: index for index, identifier in enumerate(self.image_ids)}
        if len(self.id_index) != len(images):
            raise ValueError("Coverage requires unique image IDs")
        self.image_count = len(images)
        self.flat = []
        image_of, feature_ids, classes, boxes = [], [], [], []
        self.image_objects = []
        self.protected = set()
        for i, image in enumerate(images):
            _check(cancelled)
            if image.get("protection_errors"):
                self.protected.add(i)
            owned = []
            for obj in image["objects"]:
                owned.append(len(self.flat))
                self.flat.append(obj)
                image_of.append(i)
                feature_ids.append(int(obj["feature_index"]))
                classes.append(int(obj["class_id"]))
                boxes.append(obj["box"])
            self.image_objects.append(np.asarray(owned, np.int64))
        self.object_count = len(self.flat)
        if scene is not None and (len(scene.shape) != 2 or scene.shape[0] != self.image_count or (self.image_count and scene.shape[1] < 1)):
            raise ValueError("Cached scene rows do not match the frozen source")
        if objects is not None and (len(objects.shape) != 2 or (feature_ids and (max(feature_ids) >= objects.shape[0] or min(feature_ids) < 0 or objects.shape[1] < 1))):
            raise ValueError("Cached object rows do not match the frozen source")
        if len(set(feature_ids)) != self.object_count:
            raise ValueError("Coverage object feature indices must be unique")
        self.image_of = np.asarray(image_of, np.int32)
        self.feature_ids = np.asarray(feature_ids, np.int64)
        self.classes = np.asarray(classes, np.int32)
        self.boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        self.dimensions = np.asarray([[im["width"], im["height"]] for im in images], np.float64).reshape(-1, 2)
        if np.any(self.dimensions <= 0) or not np.isfinite(self.dimensions).all():
            raise ValueError("Coverage requires valid original image dimensions")
        self.by_class = {int(c): np.flatnonzero(self.classes == c) for c in np.unique(self.classes)}
        self.class_values, self.class_local, self.class_counts = np.unique(self.classes, return_inverse=True, return_counts=True)
        self.negative = np.asarray([not im["objects"] and not im.get("protection_errors") for im in images], bool)
        self.valid_images = np.asarray([not im.get("protection_errors") for im in images], bool)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.directory / "proofs.sqlite3", timeout=60, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS proofs(kind TEXT,q INTEGER,bits BLOB,sha TEXT,PRIMARY KEY(kind,q))")
        self._db.execute("CREATE TABLE IF NOT EXISTS object_row_proofs(q INTEGER,r INTEGER,score REAL,sha TEXT,PRIMARY KEY(q,r))")
        self._db.execute("CREATE TABLE IF NOT EXISTS certificate_meta(key TEXT PRIMARY KEY,value TEXT)")
        metric = self._db.execute("SELECT value FROM certificate_meta WHERE key='metric'").fetchone()
        if metric is None:
            # Earlier generations used signed scores only on exact misses.
            # At a zero floor those incomplete old bitmaps cannot be reused.
            if config.object_threshold == 0:
                self._db.execute("DELETE FROM proofs WHERE kind='object'")
            if config.scene_threshold == 0 or config.negative_scene_threshold == 0:
                self._db.execute("DELETE FROM proofs WHERE kind='scene'")
            self._db.execute("INSERT INTO certificate_meta VALUES ('metric','clipped-cosine-v1')")
        elif metric[0] != 'clipped-cosine-v1':
            raise ValueError("Unsupported direct-coverage certificate metric")
        self._db.commit()

    def close(self):
        with self._lock:
            self._db.close()
        for attr in ("group_of", "representatives", "weights", "isolation", "object_rank_images", "object_rank_scores",
                     "object_rank_rows", "scene_rank_images", "scene_rank_scores"):
            value = getattr(self, attr, None)
            if isinstance(value, np.memmap) and not value._mmap.closed:
                value._mmap.close()

    def _features(self, rows, prefix=None):
        if self.objects is None:
            raise CoverageUnavailable("New direct coverage needs the saved cached object descriptors")
        ids = self.feature_ids[np.asarray(rows, np.int64)]
        if isinstance(self.objects, np.ndarray) and prefix is not None:
            values = np.asarray(self.objects[ids, :prefix], np.float32)
        else:
            values = np.asarray(self.objects[ids], np.float32)
            if prefix is not None:
                values = values[:, :prefix]
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Invalid cached object descriptors")
        return values

    def _scale(self, q, rows):
        """Same modern native/occupancy metric as pair_evidence, one-to-many."""
        rows = np.asarray(rows, np.int64)
        own = self.image_of[q]
        other = self.image_of[rows]
        a = self.boxes[q, 2:]
        b = self.boxes[rows, 2:]
        occupancy = np.exp(-.5 * np.abs(np.log(np.maximum(a, 1e-8) / np.maximum(b, 1e-8))).mean(axis=1))
        pa = a * self.dimensions[own]
        pb = b * self.dimensions[other]
        bbox = np.exp(-.5 * np.abs(np.log(np.maximum(pa, 1e-8) / np.maximum(pb, 1e-8))).mean(axis=1))
        image = np.exp(-.5 * np.abs(np.log(self.dimensions[own] / self.dimensions[other])).mean(axis=1))
        return np.minimum(occupancy, np.minimum(bbox, image))

    def _cosines(self, q, rows):
        query = self._features([q])[0]
        norm = float(np.linalg.norm(query))
        if norm <= 1e-8:
            raise ValueError("Zero cached object descriptor")
        values = self._features(rows)
        lengths = np.linalg.norm(values, axis=1)
        if np.any(lengths <= 1e-8):
            raise ValueError("Zero cached object descriptor")
        return np.clip((values @ query) / (lengths * norm), -1, 1)

    def _seed_row(self, q, candidates):
        own = int(self.image_of[q])
        values = {own: (1.0, q)}
        candidates = np.asarray(sorted(set(int(r) for r in candidates if int(r) != q)), np.int64)
        for start in range(0, len(candidates), BLOCK):
            rows = candidates[start:start + BLOCK]
            eligible = (self._scale(q, rows) >= self.config.rarity_scale_threshold) & self.valid_images[self.image_of[rows]]
            rows = rows[eligible]
            if not len(rows):
                continue
            for row, score in zip(rows.tolist(), self._cosines(q, rows).tolist()):
                image = int(self.image_of[row])
                score = max(0., float(score))
                previous = values.get(image)
                if previous is None or score > previous[0] or (score == previous[0] and row < previous[1]):
                    values[image] = (score, row)
        ordered = [(own, values.pop(own))] + sorted(values.items(), key=lambda item: (-item[1][0], self.image_ids[item[0]], item[1][1]))
        ordered = ordered[:RANK_K]
        self.object_rank_images[q, :len(ordered)] = [i for i, _ in ordered]
        self.object_rank_scores[q, :len(ordered)] = [v[0] for _, v in ordered]
        self.object_rank_rows[q, :len(ordered)] = [v[1] for _, v in ordered]

    def _neighbor_rows(self, class_rows, neighbors_fn, cancelled, exclude_images=True):
        n = len(class_rows)
        if n < 2:
            return np.empty((n, 0), np.int64)
        k = min(256, n - 1)
        width = int(self.objects.shape[1])
        with tempfile.TemporaryDirectory(prefix="coverage-neighbors-", dir=self.directory) as temporary:
            matrix = _array(Path(temporary) / "class.npy", (n, width), np.float32)
            for start in range(0, n, BLOCK):
                _check(cancelled)
                values = self._features(class_rows[start:start + BLOCK])
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                if np.any(norms <= 1e-8):
                    raise ValueError("Zero cached object descriptor")
                matrix[start:start + len(values)] = values / norms
            matrix.flush()
            if neighbors_fn is not None:
                try:
                    inspect.signature(neighbors_fn).bind(matrix, k, (self.image_of[class_rows] if exclude_images else np.arange(n)))
                except (TypeError, ValueError):
                    result = neighbors_fn(matrix, k)
                else:
                    result = neighbors_fn(matrix, k, self.image_of[class_rows] if exclude_images else np.arange(n))
                neighbors = result[0] if isinstance(result, tuple) else result
                neighbors = np.asarray(neighbors, np.int64)
                if neighbors.ndim != 2 or neighbors.shape[0] != n:
                    raise ValueError("Invalid class neighbor result")
                output = neighbors.copy()
            else:
                # Small fixtures and a bounded exact fallback; never allocate NxN.
                output = np.empty((n, k), np.int64)
                for start in range(0, n, 16):
                    _check(cancelled)
                    scores = np.asarray(matrix[start:start + 16] @ matrix.T)
                    for j, row in enumerate(scores):
                        row[start + j] = -np.inf
                        indices = np.argpartition(-row, k - 1)[:k]
                        output[start + j] = indices[np.lexsort((indices, -row[indices]))]
            matrix._mmap.close()
        return output

    def _group(self, neighbors, cancelled, progress):
        # Grouping is exact and independent of the ANN graph. Query tiles
        # compare with the fixed centers present at tile start; sequential
        # checks against newly created centers preserve the original order.
        group_of = np.full(self.object_count, -1, np.int32)
        representatives = []
        threshold = 1 - (1 - self.config.object_threshold) / 4
        done = 0
        cache_tag = digest(SCHEMA)[:8]
        checkpoint_path = self.directory / ("group-checkpoint-" + cache_tag + ".json")
        checkpoint = _checkpoint(checkpoint_path, self.binding)
        for cls, rows in sorted(self.by_class.items()):
            source_buckets = defaultdict(list)
            for q in rows.tolist():
                im = self.images[int(self.image_of[q])]
                source = str(im.get("source_group") or im.get("video_id") or im["image_id"])
                source_buckets[source].append(q)
            for items in source_buckets.values():
                items.sort(key=lambda q: (-float(np.prod(self.boxes[q, 2:] * self.dimensions[self.image_of[q]])), self.image_ids[self.image_of[q]], int(self.flat[q]["label_row"])))
            ordered_sources = sorted(source_buckets)
            order = [source_buckets[s][position] for position in range(max(map(len, source_buckets.values()), default=0)) for s in ordered_sources if position < len(source_buckets[s])]
            first_group = len(representatives)
            class_representatives = []
            entry = checkpoint["classes"].get(str(cls))
            processed = 0
            if entry:
                group_path = self.directory / entry["group_file"]
                representative_path = self.directory / entry["representative_file"]
                if entry["first_group"] != first_group or _file_sha(group_path) != entry["groups"] or _file_sha(representative_path) != entry["representatives"]:
                    raise ValueError("A coverage grouping checkpoint changed")
                values = np.load(group_path, allow_pickle=False)
                saved = np.load(representative_path, allow_pickle=False)
                if values.shape != (len(rows),) or not 0 <= entry["processed"] <= len(rows):
                    raise ValueError("Invalid coverage grouping checkpoint")
                group_of[rows] = values
                representatives.extend(saved.tolist())
                class_representatives = list(range(first_group, len(representatives)))
                processed = entry["processed"]
                done += processed
                _notify(progress, done, self.object_count, "groups_resumed")
                if processed == len(rows):
                    continue
            prefix_width = min(1536, int(self.objects.shape[1]))
            prefix_path = self.directory / ("anchor-prefix-" + str(cls) + ".npy")
            prefixes = _array(prefix_path, (len(rows), prefix_width), np.float32)
            remaining_norms = np.empty(len(rows), np.float64)
            try:
                # Reconstruct the linear-size prefix cache on partial resume.
                for start in range(0, len(class_representatives), BLOCK):
                    _check(cancelled)
                    refs = representatives[first_group + start:first_group + start + BLOCK]
                    values = self._features(refs).astype(np.float64)
                    values /= np.linalg.norm(values, axis=1, keepdims=True)
                    prefixes[start:start + len(values)] = values[:, :prefix_width]
                    remaining_norms[start:start + len(values)] = np.linalg.norm(values[:, prefix_width:], axis=1)

                def first_match(q, local_candidates):
                    for start in range(0, len(local_candidates), BLOCK):
                        _check(cancelled)
                        local = local_candidates[start:start + BLOCK]
                        refs = np.asarray([representatives[first_group + int(v)] for v in local], np.int64)
                        eligible = self._scale(q, refs) >= self.config.rarity_scale_threshold
                        refs, local = refs[eligible], local[eligible]
                        if len(refs):
                            passing = np.flatnonzero(self._cosines(q, refs) >= threshold)
                            if len(passing):
                                return int(local[int(passing[0])])
                    return -1

                for batch_start in range(processed, len(order), 64):
                    _check(cancelled)
                    batch_rows = order[batch_start:batch_start + 64]
                    queries = self._features(batch_rows).astype(np.float64)
                    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
                    query_remaining = np.linalg.norm(queries[:, prefix_width:], axis=1)
                    fixed_count = len(class_representatives)
                    choices = np.full(len(batch_rows), -1, np.int64)
                    for center_start in range(0, fixed_count, 4096):
                        _check(cancelled)
                        active = np.flatnonzero(choices < 0)
                        if not len(active):
                            break
                        center_stop = min(center_start + 4096, fixed_count)
                        bound = queries[active, :prefix_width] @ np.asarray(prefixes[center_start:center_stop], np.float64).T
                        bound += query_remaining[active, None] * remaining_norms[None, center_start:center_stop]
                        for local, offset in enumerate(active):
                            possible = np.flatnonzero(bound[local] + 1e-5 >= threshold) + center_start
                            choices[offset] = first_match(batch_rows[int(offset)], possible)
                    for offset, q in enumerate(batch_rows):
                        _check(cancelled)
                        chosen = int(choices[offset])
                        if chosen < 0 and len(class_representatives) > fixed_count:
                            local = np.arange(fixed_count, len(class_representatives))
                            bound = np.asarray(prefixes[local], np.float64) @ queries[offset, :prefix_width]
                            bound += remaining_norms[local] * query_remaining[offset]
                            chosen = first_match(q, local[bound + 1e-5 >= threshold])
                        if chosen < 0:
                            chosen = len(class_representatives)
                            prefixes[chosen] = queries[offset, :prefix_width]
                            remaining_norms[chosen] = query_remaining[offset]
                            representatives.append(q)
                            class_representatives.append(first_group + chosen)
                        group_of[q] = first_group + chosen
                    done += len(batch_rows)
                    processed = batch_start + len(batch_rows)
                    if processed % 1024 == 0 or processed == len(order):
                        generation = cache_tag + "-" + str(cls) + "-" + str(processed)
                        group_path = self.directory / ("group-class-" + generation + ".npy")
                        representative_path = self.directory / ("group-representatives-" + generation + ".npy")
                        _save_array(group_path, group_of[rows])
                        _save_array(representative_path, np.asarray(representatives[first_group:], np.int64))
                        checkpoint["classes"][str(cls)] = {"first_group": first_group, "processed": processed,
                            "group_file": group_path.name, "representative_file": representative_path.name,
                            "groups": _file_sha(group_path), "representatives": _file_sha(representative_path)}
                        _atomic_json(checkpoint_path, checkpoint)
                    _notify(progress, done, self.object_count, "groups")
            finally:
                prefixes._mmap.close()
                prefix_path.unlink(missing_ok=True)
        return group_of, np.asarray(representatives, np.int64)

    def _exact_rarity_means(self, rows, graph, cancelled, progress, cls):
        """Exact nearest-five distances; ANN supplies an initial bound only.

        Batched full-vector fp32 dots conservatively screen EVERY exemplar;
        original-vector fp64 comparisons resolve every possible improvement.
        Completed independent query tiles are restart-safe cached work.
        """
        count = len(rows)
        keep = min(5, count - 1)
        width = int(self.objects.shape[1])
        cache_tag = digest(SCHEMA)[:8]
        checkpoint_path = self.directory / ("rarity-blocks-" + cache_tag + "-" + str(cls) + ".json")
        binding = {"source": self.binding, "rarity": EXACT_RARITY, "width": width,
                   "rows": hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()}
        checkpoint = _checkpoint(checkpoint_path, binding)
        means = np.empty(count, np.float64)
        processed = 0
        for entry in checkpoint.get("blocks", []):
            path = self.directory / entry["file"]
            if entry["start"] != processed or not processed < entry["stop"] <= count or _file_sha(path) != entry["sha"]:
                raise ValueError("An exact-rarity query checkpoint changed")
            values = np.load(path, allow_pickle=False)
            if values.shape != (entry["stop"] - processed,) or not np.isfinite(values).all():
                raise ValueError("Invalid exact-rarity query checkpoint")
            means[processed:entry["stop"]] = values
            processed = entry["stop"]
        if processed:
            _notify(progress, processed, count, "exact_rarity_class_" + str(cls) + "_resumed")
        if processed == count:
            return means
        error = _cosine_screen_error(width)
        with tempfile.TemporaryDirectory(prefix="coverage-rarity-", dir=self.directory) as temporary, ExitStack() as cleanup:
            matrix = _array(Path(temporary) / "values.npy", (count, width), np.float32)
            cleanup.callback(matrix._mmap.close)
            norms = np.empty(count, np.float64)
            for start in range(0, count, BLOCK):
                _check(cancelled)
                values = self._features(rows[start:start + BLOCK])
                lengths = np.linalg.norm(values.astype(np.float64), axis=1)
                if np.any(lengths <= 1e-8):
                    raise ValueError("Zero cached object descriptor")
                end = start + len(values)
                matrix[start:end] = values
                norms[start:end] = lengths
            matrix.flush()
            for start in range(processed, count, 256):
                _check(cancelled)
                stop = min(start + 256, count)
                queries32 = np.asarray(matrix[start:stop], np.float32)
                queries = np.asarray(matrix[start:stop], np.float64)
                nearest = []
                selected = []
                for offset, local in enumerate(range(start, stop)):
                    candidates = sorted(set(int(v) for v in graph[local] if 0 <= int(v) < count and int(v) != local))
                    if len(candidates) < keep:
                        candidates = [v for v in range(count) if v != local][:keep]
                    targets = np.asarray(candidates, np.int64)
                    scores = np.clip((np.asarray(matrix[targets], np.float64) @ queries[offset]) /
                                     (norms[targets] * norms[local]), -1, 1)
                    order = np.lexsort((targets, -scores))[:keep]
                    nearest.append(scores[order]); selected.append(targets[order])
                for target_start in range(0, count, 4096):
                    _check(cancelled)
                    target_stop = min(target_start + 4096, count)
                    with np.errstate(over="ignore", invalid="ignore"):
                        approximate = queries32 @ np.asarray(matrix[target_start:target_stop], np.float32).T
                        bounds = approximate / (norms[start:stop, None] * norms[None, target_start:target_stop])
                    bounds[~np.isfinite(bounds)] = np.inf
                    for offset, local in enumerate(range(start, stop)):
                        possible = np.flatnonzero(bounds[offset] + error >= float(nearest[offset][-1])) + target_start
                        possible = possible[(possible != local) & ~np.isin(possible, selected[offset])]
                        for batch in range(0, len(possible), BLOCK):
                            targets = possible[batch:batch + BLOCK]
                            scores = np.clip((np.asarray(matrix[targets], np.float64) @ queries[offset]) /
                                             (norms[targets] * norms[local]), -1, 1)
                            all_scores = np.concatenate((nearest[offset], scores))
                            all_targets = np.concatenate((selected[offset], targets))
                            order = np.lexsort((all_targets, -all_scores))[:keep]
                            nearest[offset], selected[offset] = all_scores[order], all_targets[order]
                for offset, scores in enumerate(nearest):
                    means[start + offset] = np.sqrt(np.maximum(0., 2 * (1 - scores))).mean()
                path = self.directory / ("rarity-means-" + cache_tag + "-" + str(cls) + "-" + str(start) + "-" + str(stop) + ".npy")
                _save_array(path, means[start:stop])
                checkpoint.setdefault("blocks", []).append({"start": start, "stop": stop, "file": path.name, "sha": _file_sha(path)})
                _atomic_json(checkpoint_path, checkpoint)
                _notify(progress, stop, count, "exact_rarity_class_" + str(cls))
        return means

    def _rarity_seed_graph(self, representatives, groups, group_of):
        local_of = np.full(len(representatives), -1, np.int64)
        local_of[groups] = np.arange(len(groups))
        graph = np.full((len(groups), RANK_K), -1, np.int32)
        for local, group in enumerate(groups):
            rows = self.object_rank_rows[int(representatives[group])]
            candidates = np.unique(group_of[rows[rows >= 0]])
            candidates = candidates[(candidates != group) & (local_of[candidates] >= 0)]
            graph[local, :len(candidates)] = local_of[candidates]
        return graph

    def _rarity(self, representatives, cancelled, neighbors_fn, progress=None, group_of=None):
        group_classes = self.classes[representatives]
        counts = {int(c): int(np.count_nonzero(group_classes == c)) for c in np.unique(group_classes)}
        median = float(np.median(list(counts.values()))) if counts else 1.
        weights = np.empty(len(representatives), np.float64)
        isolation = np.empty(len(representatives), np.float64)
        cache_tag = digest(SCHEMA)[:8]
        checkpoint_path = self.directory / ("rarity-checkpoint-" + cache_tag + ".json")
        checkpoint = _checkpoint(checkpoint_path, {"source": self.binding, "rarity": EXACT_RARITY,
                    "representatives": hashlib.sha256(np.asarray(representatives, np.int64).tobytes()).hexdigest()})
        for cls, count in sorted(counts.items()):
            _check(cancelled)
            groups = np.flatnonzero(group_classes == cls)
            if count > 1:
                entry = checkpoint["classes"].setdefault(str(cls), {})
                distance_path = self.directory / ("rarity-distances-" + cache_tag + "-" + str(cls) + ".npy")
                if entry.get("distances"):
                    if not distance_path.is_file() or _file_sha(distance_path) != entry["distances"]:
                        raise ValueError("An exact-rarity checkpoint changed")
                    distances = np.load(distance_path, allow_pickle=False)
                    if distances.shape != (count,):
                        raise ValueError("Invalid exact-rarity checkpoint shape")
                    _notify(progress, count, count, "exact_rarity_class_" + str(cls) + "_resumed")
                else:
                    graph_path = self.directory / ("rarity-neighbors-" + cache_tag + "-" + str(cls) + ".npy")
                    if entry.get("neighbors"):
                        if not graph_path.is_file() or _file_sha(graph_path) != entry["neighbors"]:
                            raise ValueError("A rarity-neighbor checkpoint changed")
                        graph = np.load(graph_path, mmap_mode="r", allow_pickle=False)
                    else:
                        if group_of is not None and hasattr(self, "object_rank_rows"):
                            _notify(progress, 0, count, "rarity_reusing_frozen_neighbors_class_" + str(cls))
                            graph = self._rarity_seed_graph(representatives, groups, group_of)
                        else:
                            _notify(progress, 0, count, "rarity_neighbors_class_" + str(cls))
                            graph = self._neighbor_rows(representatives[groups], neighbors_fn, cancelled, exclude_images=False)
                        _check(cancelled)
                        _save_array(graph_path, np.asarray(graph, np.int32))
                        entry["neighbors"] = _file_sha(graph_path)
                        _atomic_json(checkpoint_path, checkpoint)
                    distances = self._exact_rarity_means(representatives[groups], graph, cancelled, progress, cls)
                    if isinstance(graph, np.memmap):
                        graph._mmap.close()
                    _check(cancelled)
                    _save_array(distance_path, distances)
                    entry["distances"] = _file_sha(distance_path)
                    _atomic_json(checkpoint_path, checkpoint)
                percentiles = _midranks(distances)
            else:
                percentiles = np.ones(1)
            isolation[groups] = percentiles
            weights[groups] = np.clip(math.sqrt(median / count), .5, 2.) * (1 + percentiles)
        return weights, isolation

    def _rank_digest(self, rows):
        value = hashlib.sha256()
        for array in (self.object_rank_images, self.object_rank_scores, self.object_rank_rows):
            for start in range(0, len(rows), BLOCK):
                value.update(np.asarray(array[rows[start:start + BLOCK]]).tobytes())
        return value.hexdigest()

    def _publish(self, group_of, representatives, weights, isolation, cancelled):
        staged = {}
        for name, values in (("group-of", group_of), ("representatives", representatives), ("weights", weights), ("isolation", isolation)):
            path = self.directory / (name + ".npy.pending")
            _save_array(path, values)
            staged[name] = path
        _check(cancelled)
        artifacts = {name + ".npy": _file_sha(staged.get(name, self.directory / (name + ".npy"))) for name in FROZEN_ARRAYS}
        manifest = {"schema": SCHEMA, "artifacts": artifacts, "binding": self.binding, "image_ids": self.image_ids,
                    "object_count": self.object_count, "group_count": len(representatives),
                    "grouping_radius": .5 * math.sqrt(2 * (1 - self.config.object_threshold)),
                    "rarity": EXACT_RARITY, "ranking": "frozen64_image_shortlist_proxy", "complete": True}
        manifest["accounting_digest"] = digest({**manifest, "group_sha": hashlib.sha256(np.asarray(group_of).tobytes()).hexdigest(), "weights_sha": hashlib.sha256(np.asarray(weights).tobytes()).hexdigest()})
        _check(cancelled)
        # Cancellation cannot strand a previously complete generation between
        # replacing its arrays and publishing their new integrity manifest.
        for name, path in staged.items():
            os.replace(path, self.directory / (name + ".npy"))
        _atomic_json(self.directory / "manifest.json", manifest)
        return manifest

    def _build(self, neighbors_fn, cancelled, progress):
        if self.scene is None or self.objects is None:
            raise CoverageUnavailable("Coverage preparation requires cached scene and object features")
        checkpoint_path = self.directory / "prepare-checkpoint.json"
        checkpoint = _checkpoint(checkpoint_path, self.binding)
        shape = (self.object_count, RANK_K)
        resumed_arrays = True
        for attr, dtype, fill in (("object_rank_images", np.int32, -1), ("object_rank_scores", np.float32, 0), ("object_rank_rows", np.int32, -1)):
            path = self.directory / (attr.replace("_", "-") + ".npy")
            if path.is_file() and checkpoint_path.is_file():
                array = np.load(path, mmap_mode="r+", allow_pickle=False)
                if array.shape != shape or array.dtype != dtype:
                    raise ValueError("Invalid coverage ranking checkpoint shape")
            else:
                array = _array(path, shape, dtype, fill)
                resumed_arrays = False
            setattr(self, attr, array)
        if not resumed_arrays:
            for entry in checkpoint["classes"].values():
                entry.pop("ranking", None)
        _atomic_json(checkpoint_path, checkpoint)
        neighbors = {}
        for cls, rows in sorted(self.by_class.items()):
            _check(cancelled)
            graph_path = self.directory / ("neighbors-" + str(cls) + ".npy")
            entry = checkpoint["classes"].setdefault(str(cls), {})
            if entry.get("neighbors"):
                if not graph_path.is_file() or _file_sha(graph_path) != entry["neighbors"]:
                    raise ValueError("A coverage-neighbor checkpoint changed")
                graph = np.load(graph_path, mmap_mode="r", allow_pickle=False)
                _notify(progress, len(rows), len(rows), "neighbors_class_" + str(cls) + "_resumed")
            else:
                _notify(progress, 0, len(rows), "neighbors_class_" + str(cls))
                graph = self._neighbor_rows(rows, neighbors_fn, cancelled)
                _check(cancelled)
                _save_array(graph_path, np.asarray(graph, np.int32))
                entry["neighbors"] = _file_sha(graph_path)
                _atomic_json(checkpoint_path, checkpoint)
                graph = np.load(graph_path, mmap_mode="r", allow_pickle=False)
            neighbors[cls] = graph
            if entry.get("ranking"):
                if self._rank_digest(rows) != entry["ranking"]:
                    raise ValueError("A coverage ranking checkpoint changed")
                _notify(progress, len(rows), len(rows), "object_ranking_class_" + str(cls) + "_resumed")
            else:
                # An interrupted class is rewritten; completed earlier classes
                # and their expensive source-bound ANN builds remain reusable.
                self.object_rank_images[rows] = -1
                self.object_rank_scores[rows] = 0
                self.object_rank_rows[rows] = -1
                for j, q in enumerate(rows):
                    _check(cancelled)
                    valid = graph[j][(graph[j] >= 0) & (graph[j] < len(rows))]
                    self._seed_row(int(q), rows[valid])
                    if j % 256 == 0:
                        _notify(progress, j, len(rows), "object_ranking_class_" + str(cls))
                for array in (self.object_rank_images, self.object_rank_scores, self.object_rank_rows):
                    array.flush()
                entry["ranking"] = self._rank_digest(rows)
                _atomic_json(checkpoint_path, checkpoint)
                _notify(progress, len(rows), len(rows), "object_ranking_class_" + str(cls))
        if checkpoint.get("groups"):
            for name, expected in checkpoint["groups"].items():
                if _file_sha(self.directory / (name + ".npy")) != expected:
                    raise ValueError("A coverage grouping checkpoint changed")
            group_of = np.load(self.directory / "group-of.npy", allow_pickle=False)
            representatives = np.load(self.directory / "representatives.npy", allow_pickle=False)
            _notify(progress, self.object_count, self.object_count, "groups_resumed")
        else:
            group_of, representatives = self._group(neighbors, cancelled, progress)
            _save_array(self.directory / "group-of.npy", group_of)
            _save_array(self.directory / "representatives.npy", representatives)
            checkpoint["groups"] = {name: _file_sha(self.directory / (name + ".npy")) for name in ("group-of", "representatives")}
            _atomic_json(checkpoint_path, checkpoint)
        for cls, graph in neighbors.items():
            graph._mmap.close()
        del neighbors
        weights, isolation = self._rarity(representatives, cancelled, neighbors_fn, progress, group_of)
        self.scene_rank_images = _array(self.directory / "scene-rank-images.npy", (self.image_count, RANK_K), np.int32, -1)
        self.scene_rank_scores = _array(self.directory / "scene-rank-scores.npy", (self.image_count, RANK_K), np.float32, 0)
        id_order = np.argsort(np.argsort(np.asarray(self.image_ids), kind="stable"), kind="stable")
        missing_scene = self._missing_scene_queries()
        scene_norms = self._scene_norms(cancelled) if missing_scene else None
        for start in range(0, self.image_count, 32):
            _check(cancelled)
            query = np.asarray(self.scene[start:start + 32], np.float32)
            if not np.isfinite(query).all():
                raise ValueError("Invalid cached scene descriptors")
            with np.errstate(over="ignore", invalid="ignore"):
                raw_scores = query @ np.asarray(self.scene).T
            needed = [i for i in range(start, min(start + 32, self.image_count)) if i in missing_scene]
            if needed:
                local = np.asarray(needed, np.int64) - start
                bitmaps, _ = self._scene_bitmaps(np.asarray(needed, np.int64), raw_scores[local], scene_norms, cancelled)
                self._save_scene_batch(needed, bitmaps, cancelled)
            scores = np.clip(raw_scores, 0, 1)
            for local, values in enumerate(scores):
                i = start + local
                valid = self.valid_images & ((not self.negative[i]) | self.negative)
                values[~valid] = -np.inf
                values[i] = 1
                ordered = np.lexsort((id_order, -values))
                targets = ordered[(ordered != i) & np.isfinite(values[ordered])][:RANK_K - 1]
                targets = np.concatenate(([i], targets))
                self.scene_rank_images[i, :len(targets)] = targets
                self.scene_rank_scores[i, :len(targets)] = values[targets]
            _notify(progress, min(start + 32, self.image_count), self.image_count, "scene_ranking")
        for array in [self.object_rank_images, self.object_rank_scores, self.object_rank_rows, self.scene_rank_images, self.scene_rank_scores]:
            array.flush()
        self._publish(group_of, representatives, weights, isolation, cancelled)

    def _load(self, manifest):
        if manifest.get("schema") == SCHEMA and manifest.get("rarity") != EXACT_RARITY:
            raise ValueError("Saved coverage lacks exact source-wide rarity; recompute its rarity generation")
        if not manifest.get("artifacts"):
            raise ValueError("Saved coverage lacks frozen-artifact integrity bindings; prepare a new generation")
        self._artifact_states = {}
        for name, expected in manifest["artifacts"].items():
            path = self.directory / name
            before = _state(path)
            if _file_sha(path) != expected or _state(path) != before:
                raise ValueError("A frozen coverage artifact changed")
            self._artifact_states[name] = before
        self._manifest_state = _state(self.directory / "manifest.json")
        self.accounting_digest = manifest["accounting_digest"]
        for attr, file in [("group_of", "group-of"), ("representatives", "representatives"), ("weights", "weights"), ("isolation", "isolation"), ("object_rank_images", "object-rank-images"), ("object_rank_scores", "object-rank-scores"), ("object_rank_rows", "object-rank-rows"), ("scene_rank_images", "scene-rank-images"), ("scene_rank_scores", "scene-rank-scores")]:
            setattr(self, attr, np.load(self.directory / (file + ".npy"), mmap_mode="r", allow_pickle=False))
        self.group_count = len(self.representatives)
        self.total_weight = float(self.weights.sum())
        self.group_sizes = np.bincount(self.group_of, minlength=self.group_count)
        self.object_weights = (self.weights[self.group_of] / self.group_sizes[self.group_of] / self.total_weight) if self.object_count else np.empty(0)
        self._group_members = np.argsort(self.group_of, kind="stable")
        self._group_offsets = np.concatenate(([0], np.cumsum(self.group_sizes)))

    def _validate(self):
        try:
            if _state(self.directory / "manifest.json") != self._manifest_state:
                raise ValueError("Frozen coverage manifest changed")
            if any(_state(self.directory / name) != state for name, state in self._artifact_states.items()):
                raise ValueError("A frozen coverage artifact changed during this operation")
        except OSError as error:
            raise ValueError("A frozen coverage artifact is unavailable") from error

    def validate(self):
        self._validate()
        return True

    @property
    def proof_generation(self):
        # Appending a reusable proof is cache warming, not a new scientific or
        # review generation. Unrelated rejected candidates cannot stale plans.
        return digest({"accounting": self.accounting_digest, "certificate_schema": SCHEMA,
                       "qualification": QUALIFICATION})

    def _proof(self, kind, q):
        with self._lock:
            row = self._db.execute("SELECT bits,sha FROM proofs WHERE kind=? AND q=?", (kind, int(q))).fetchone()
        if row is None:
            return None
        return self._decode_proof(row[0], row[1])

    def _decode_proof(self, compressed, sha):
        try:
            raw = zlib.decompress(compressed)
        except zlib.error as error:
            raise ValueError("A direct-coverage certificate is corrupt") from error
        if hashlib.sha256(raw).hexdigest() != sha or len(raw) != (self.image_count + 7) // 8:
            raise ValueError("A direct-coverage certificate is corrupt")
        return np.frombuffer(raw, np.uint8)

    def _missing_scene_queries(self):
        # Decode one disk-backed bitmap at a time. Keeping all compressed
        # N-bit rows simultaneously could otherwise require quadratic RAM.
        present = set()
        with self._lock:
            for q, compressed, sha in self._db.execute("SELECT q,bits,sha FROM proofs WHERE kind='scene'"):
                if not 0 <= q < self.image_count:
                    raise ValueError("A scene certificate has an invalid source row")
                self._decode_proof(compressed, sha)
                present.add(q)
        return set(range(self.image_count)) - present

    def _scene_norms(self, cancelled):
        if self.scene is None:
            raise CoverageUnavailable("New scene coverage needs the saved cached scene descriptors")
        norms = np.empty(self.image_count, np.float64)
        for start in range(0, self.image_count, BLOCK):
            _check(cancelled)
            values = np.asarray(self.scene[start:start + BLOCK], np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Invalid cached scene descriptors")
            with np.errstate(over="ignore", invalid="ignore"):
                lengths = np.linalg.norm(values, axis=1)
            tiny = np.finfo(self.scene.dtype).tiny if np.issubdtype(self.scene.dtype, np.floating) else np.finfo(np.float64).tiny
            absolute = np.abs(values)
            unsafe = ((lengths == 0) & np.any(absolute > 0, axis=1)) | np.any((absolute > 0) & (absolute < tiny), axis=1)
            # Subnormal inputs / underflowed norm reductions have no useful
            # relative-error certificate. Force the authoritative GEMV path.
            lengths[unsafe] = np.inf
            norms[start:start + len(values)] = lengths
        return norms

    def _scene_bitmaps(self, queries, raw_scores, target_norms, cancelled):
        """Certify GEMM scores against the existing 256-target GEMV metric.

        BOTH dot products can round. Twice gamma_(2D), with conservative fp64
        norm-rounding allowance, bounds their difference times original row
        norms. No unit-length assumption or descriptor normalization is used.
        Every uncertain global target block is recomputed with exactly the
        old query cast, original target dtype/order, and GEMV block shape.
        """
        query = np.asarray(self.scene[queries], np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            query_norms = np.linalg.norm(query.astype(np.float64), axis=1)
            absolute = np.abs(query.astype(np.float64))
            query_norms[np.any((absolute > 0) & (absolute < np.finfo(np.float32).tiny), axis=1)] = np.inf
            norm_products = query_norms[:, None] * target_norms[None, :]
            gamma = _cosine_screen_error(self.scene.shape[1])
            error = (2 * gamma) * norm_products
            error += 4 * self.scene.shape[1] * np.finfo(np.float32).tiny
            # A finite GEMM result does not imply the old GEMV avoided
            # intermediate overflow/cancellation. Cauchy bounds every partial
            # absolute sum by this norm product; retain generous headroom in
            # the ORIGINAL promoted accumulator dtype, or refine instead.
            accumulator = np.result_type(self.scene.dtype, np.float32)
            safe_limit = np.finfo(accumulator).max / (4 * (1 + gamma))
            error[norm_products > safe_limit] = np.inf
            lower = np.clip(raw_scores - error, 0, 1)
            upper = np.clip(raw_scores + error, 0, 1)
        thresholds = np.where(self.negative[queries], self.config.negative_scene_threshold, self.config.scene_threshold)
        finite = np.isfinite(raw_scores) & np.isfinite(error)
        positive = finite & (lower >= thresholds[:, None])
        negative = finite & (upper < thresholds[:, None])
        valid = self.valid_images[None, :] & ((~self.negative[queries])[:, None] | self.negative[None, :])
        covered = positive & valid
        uncertain = ~(positive | negative) & valid
        uncertain[np.arange(len(queries)), queries] = False
        refined = 0
        for local, q in enumerate(queries):
            _check(cancelled)
            for start in range(0, self.image_count, BLOCK):
                if not np.any(uncertain[local, start:start + BLOCK]):
                    continue
                _check(cancelled)
                targets = np.arange(start, min(start + BLOCK, self.image_count))
                with np.errstate(over="ignore", invalid="ignore"):
                    scores = np.clip(np.asarray(self.scene[targets]) @ query[local], 0, 1)
                covered[local, targets] = (scores >= thresholds[local]) & valid[local, targets]
                refined += 1
        covered[np.arange(len(queries)), queries] = True
        return covered, refined

    def _save_scene_batch(self, queries, covered, cancelled):
        values = []
        for q, bitmap in zip(queries, covered):
            raw = np.packbits(bitmap, bitorder="little").tobytes()
            values.append(('scene', int(q), zlib.compress(raw), hashlib.sha256(raw).hexdigest()))
        _check(cancelled)
        with self._lock:
            try:
                # Acquire the write reservation before reading existing rows;
                # another WAL writer can safely finish while this batch waits.
                self._db.execute('BEGIN IMMEDIATE')
                for _, q, _, sha in values:
                    existing = self._db.execute("SELECT sha FROM proofs WHERE kind='scene' AND q=?", (q,)).fetchone()
                    if existing is not None and existing[0] != sha:
                        raise ValueError("Conflicting exact scene certificates")
                before = self._db.total_changes
                self._db.executemany('INSERT OR IGNORE INTO proofs VALUES (?,?,?,?)', values)
                inserted = self._db.total_changes - before
                self._db.commit()
                return inserted
            except BaseException:
                self._db.rollback()
                raise

    def prewarm_scene_proofs(self, *, cancelled=None, progress=None, missing_only=True):
        """Fill exact scene certificates without touching frozen accounting.

        Safe to run beside another planner through SQLite WAL. Completed
        batches remain reusable after cancellation. With no scene vectors,
        already-complete caches are readable; missing comparisons never encode.
        """
        self._validate()
        _check(cancelled)
        missing = self._missing_scene_queries()
        queries = sorted(missing if missing_only else range(self.image_count))
        result = {"scene_queries": self.image_count, "already_cached": self.image_count - len(missing),
                  "computed_queries": 0, "inserted": 0, "refined_target_blocks": 0}
        if not queries:
            self._validate()
            return result
        norms = self._scene_norms(cancelled)
        for start in range(0, len(queries), 32):
            _check(cancelled)
            selected = np.asarray(queries[start:start + 32], np.int64)
            query = np.asarray(self.scene[selected], np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                raw_scores = query @ np.asarray(self.scene).T
            covered, refined = self._scene_bitmaps(selected, raw_scores, norms, cancelled)
            result['inserted'] += self._save_scene_batch(selected, covered, cancelled)
            result['computed_queries'] += len(selected)
            result['refined_target_blocks'] += refined
            _notify(progress, min(start + 32, len(queries)), len(queries), 'scene_proofs')
        self._validate()
        return result

    def _save_proof(self, kind, q, covered):
        raw = np.packbits(np.asarray(covered, bool), bitorder="little").tobytes()
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO proofs VALUES (?,?,?,?)", (kind, int(q), zlib.compress(raw), hashlib.sha256(raw).hexdigest()))
        return np.frombuffer(raw, np.uint8)

    def _exact(self, kind, q, cancelled):
        cached = self._proof(kind, q)
        if cached is not None:
            return cached
        covered = np.zeros(self.image_count, bool)
        if kind == "object":
            rows = self.by_class[int(self.classes[q])]
            for start in range(0, len(rows), BLOCK):
                _check(cancelled)
                target = rows[start:start + BLOCK]
                target = target[self.valid_images[self.image_of[target]] & (self._scale(q, target) >= self.config.rarity_scale_threshold)]
                if len(target):
                    scores = np.clip(self._cosines(q, target), 0, 1)
                    covered[self.image_of[target[scores >= self.config.object_threshold]]] = True
            covered[self.image_of[q]] = True
        else:
            if self.scene is None:
                raise CoverageUnavailable("New scene coverage needs the saved cached scene descriptors")
            threshold = self.config.negative_scene_threshold if self.negative[q] else self.config.scene_threshold
            query = np.asarray(self.scene[q], np.float32)
            for start in range(0, self.image_count, BLOCK):
                _check(cancelled)
                target = np.arange(start, min(start + BLOCK, self.image_count))
                scores = np.clip(np.asarray(self.scene[target]) @ query, 0, 1)
                covered[target] = (scores >= threshold) & self.valid_images[target] & ((not self.negative[q]) | self.negative[target])
            covered[q] = True
        _check(cancelled)
        return self._save_proof(kind, q, covered)

    def _object_row_scores(self, q, rows, cancelled=None, verify=True):
        """Certify selected correspondences using the full proof's block shape.

        Recomputing just a selected row changes fp32 reduction rounding. Read
        its original 256-row class block, apply the same validity/scale mask,
        then retain only the requested scores. These sparse certificates avoid
        a full-class scan for ordinary positive matches; they cannot certify
        absence. Frozen ANN scores are used for ranking only.
        """
        rows = sorted(set(map(int, rows)))
        result = {}
        for start in range(0, len(rows), BLOCK):
            _check(cancelled)
            selected = rows[start:start + BLOCK]
            placeholders = ",".join("?" for _ in selected)
            with self._lock:
                cached = self._db.execute(
                    "SELECT r,score,sha FROM object_row_proofs WHERE q=? AND r IN (" + placeholders + ")",
                    (int(q), *selected)).fetchall()
            for row, score, sha in cached:
                if digest([int(q), row, score]) != sha:
                    raise ValueError("A direct object correspondence certificate is corrupt")
                result[row] = score
        missing = set(rows) - result.keys()
        if not missing or not verify:
            return result
        if self.objects is None:
            raise CoverageUnavailable("New direct coverage needs the saved cached object descriptors")
        class_rows = self.by_class[int(self.classes[q])]
        blocks = defaultdict(list)
        for row in sorted(missing):
            if not 0 <= row < self.object_count or self.classes[row] != self.classes[q]:
                raise ValueError("Direct object correspondence requires a same-class row")
            position = int(np.searchsorted(class_rows, row))
            blocks[position // BLOCK].append(row)
        pending = []
        for block, requested in sorted(blocks.items()):
            _check(cancelled)
            target = class_rows[block * BLOCK:(block + 1) * BLOCK]
            target = target[self.valid_images[self.image_of[target]] &
                            (self._scale(q, target) >= self.config.rarity_scale_threshold)]
            scores = np.clip(self._cosines(q, target), 0, 1) if len(target) else []
            values = dict(zip(target.tolist(), scores))
            for row in requested:
                score = float(values[row]) if row in values else None
                # NaN never qualifies in the authoritative comparison. Store
                # that result as null so JSON/SQLite preserve it consistently.
                if score is not None and not math.isfinite(score):
                    score = None
                result[row] = score
                pending.append((int(q), row, score, digest([int(q), row, score])))
        _check(cancelled)
        with self._lock, self._db:
            self._db.executemany("INSERT OR REPLACE INTO object_row_proofs VALUES (?,?,?,?)", pending)
        return result

    def _witness(self, kind, q, retained, cancelled, verify=True):
        if kind == "object":
            ids, scores = self.object_rank_images[q], self.object_rank_scores[q]
            threshold = self.config.object_threshold
        else:
            ids, scores = self.scene_rank_images[q], self.scene_rank_scores[q]
            threshold = self.config.negative_scene_threshold if self.negative[q] else self.config.scene_threshold
        eligible = (ids >= 0) & (scores >= threshold)
        positions = np.flatnonzero(eligible & retained[np.maximum(ids, 0)])
        # A full certificate takes precedence even when a shortlist's score
        # rounds above the threshold. Never turn an exact negative positive.
        proof = self._proof(kind, q)
        if proof is None and kind == "object":
            for position in positions:
                image = int(ids[position])
                if image == int(self.image_of[q]):
                    return image
                row = int(self.object_rank_rows[q, position])
                score = self._object_row_scores(q, [row], cancelled, verify).get(row)
                if score is not None and score >= threshold:
                    return image
        if proof is None:
            if not verify:
                return -2
            proof = self._exact(kind, q, cancelled)
        for position in positions:
            image = int(ids[position])
            if proof[image // 8] & (1 << (image % 8)):
                return image
        intersection = np.bitwise_and(proof, np.packbits(retained, bitorder="little"))
        nonzero = np.flatnonzero(intersection)
        if not len(nonzero):
            return -1
        byte = int(nonzero[0]); value = int(intersection[byte]); bit = (value & -value).bit_length() - 1
        return byte * 8 + bit

    def _indices(self, values):
        result = set()
        for value in values:
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                index = int(value)
            else:
                if str(value) not in self.id_index:
                    raise ValueError("Unknown coverage image ID")
                index = self.id_index[str(value)]
            if not 0 <= index < self.image_count:
                raise ValueError("Unknown coverage image index")
            result.add(index)
        return result

    def _settings(self, settings):
        result = {"policy": "budgeted_coverage", "object_weight": self.config.object_weight,
                  "max_removed_images": int(self.image_count * .2), "max_affected_appearance_groups": 0,
                  "max_lost_appearance_weight_fraction": 0.}
        result.update(settings or {})
        if result["policy"] != "budgeted_coverage":
            raise ValueError("Coverage engine requires budgeted_coverage policy")
        for key in ["max_removed_images", "max_affected_appearance_groups"]:
            if type(result[key]) is not int or result[key] < 0:
                raise ValueError("Invalid coverage setting " + key)
        for key in ["object_weight", "max_lost_appearance_weight_fraction"]:
            value = result[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid coverage setting " + key)
        return result

    def _rank_coverage(self, retained):
        object_sum = 0.
        for start in range(0, self.object_count, BLOCK):
            ids = self.object_rank_images[start:start + BLOCK]
            scores = self.object_rank_scores[start:start + BLOCK]
            maximum = np.where((ids >= 0) & retained[np.maximum(ids, 0)], scores, 0).max(axis=1)
            object_sum += float(maximum @ self.object_weights[start:start + len(maximum)])
        ids = self.scene_rank_images
        scene_values = np.where((ids >= 0) & retained[np.maximum(ids, 0)], self.scene_rank_scores, 0).max(axis=1)
        return object_sum if self.object_count else 1., float(scene_values.mean()) if self.image_count else 1.

    def _scene_similarity(self, q, witness):
        positions = np.flatnonzero(self.scene_rank_images[q] == witness)
        if len(positions):
            return float(self.scene_rank_scores[q, int(positions[0])])
        if self.scene is None:
            return None
        return float(np.clip(np.asarray(self.scene[q]) @ np.asarray(self.scene[witness]), 0, 1))

    def _initial_ranking_costs(self):
        if not hasattr(self, "_initial_object_costs"):
            objects = np.zeros(self.image_count, np.float64)
            scenes = np.zeros(self.image_count, np.float64)
            # Compute each component once with the same row order/dot product
            # as ranking_delta. Slider changes only recombine these constants.
            for i, rows in enumerate(self.image_objects):
                if len(rows):
                    ids = self.object_rank_images[rows]; scores = self.object_rank_scores[rows]
                    valid = ids >= 0
                    before = np.where(valid, scores, 0).max(axis=1)
                    after = np.where(valid & (ids != i), scores, 0).max(axis=1)
                    objects[i] = float((before - after) @ self.object_weights[rows])
                ids = self.scene_rank_images[i]; scores = self.scene_rank_scores[i]
                scenes[i] = float((np.where(ids >= 0, scores, 0).max() - np.where((ids >= 0) & (ids != i), scores, 0).max()) / max(1, self.image_count))
            self._initial_object_costs, self._initial_scene_costs = objects, scenes
        return self._initial_object_costs, self._initial_scene_costs

    def _evaluate(self, retained, cancelled, verify, progress=None):
        retained = self._indices(retained)
        mask = np.zeros(self.image_count, bool); mask[list(retained)] = True
        witnesses = self.image_of.copy()
        pending = np.flatnonzero(~mask[self.image_of])
        for start in range(0, len(pending), BLOCK):
            _check(cancelled)
            rows = pending[start:start + BLOCK]
            for q in rows:
                witnesses[q] = self._witness("object", int(q), mask, cancelled, verify)
            _notify(progress, min(start + BLOCK, len(pending)), len(pending), "verify_objects")
        scene_witnesses = {}
        scene_unknown, scene_missing = [], []
        removed = sorted(set(range(self.image_count)) - retained)
        for i in removed:
            _check(cancelled)
            witness = self._witness("scene", i, mask, cancelled, verify)
            if witness >= 0:
                scene_witnesses[self.image_ids[i]] = {"witness_id": self.image_ids[witness], "witness_index": witness, "similarity": self._scene_similarity(i, witness)}
            elif witness == -2:
                scene_unknown.append(i)
            else:
                scene_missing.append(i)
        lost = np.zeros(self.group_count, bool); uncertain = np.zeros(self.group_count, bool)
        if self.object_count:
            lost[self.group_of[witnesses == -1]] = True
            uncertain[self.group_of[witnesses == -2]] = True
        uncertain &= ~lost
        worst = lost | uncertain
        object_cov, scene_cov = self._rank_coverage(mask)
        source_counts = {int(cls): int(count) for cls, count in zip(self.class_values, self.class_counts)}
        counts = np.bincount(self.class_local[mask[self.image_of]], minlength=len(self.class_values))
        retained_counts = {int(cls): int(count) for cls, count in zip(self.class_values, counts) if count}
        metrics = {"policy": "budgeted_coverage", "accounting_digest": self.accounting_digest,
                   "proof_generation": self.proof_generation, "removed_indices": removed, "retained_indices": sorted(retained),
                   "removed_ids": [self.image_ids[i] for i in removed], "retained_ids": [self.image_ids[i] for i in sorted(retained)],
                   "removed_count": len(removed), "removed_percent": 100 * len(removed) / max(1, self.image_count),
                   "affected_appearance_group_count": int(lost.sum()), "unknown_appearance_group_count": int(uncertain.sum()),
                   "affected_appearance_group_count_upper": int(worst.sum()), "appearance_group_count": self.group_count,
                   "lost_appearance_weight_fraction": float(self.weights[lost].sum() / self.total_weight) if self.total_weight else 0.,
                   "lost_appearance_weight_fraction_upper": float(self.weights[worst].sum() / self.total_weight) if self.total_weight else 0.,
                   "appearance_weight_total": self.total_weight, "total_appearance_weight": self.total_weight, "uncovered_object_count": int(np.count_nonzero(witnesses == -1)),
                   "unknown_object_count": int(np.count_nonzero(witnesses == -2)),
                   "removed_annotation_count": int(sum(len(self.images[i]["objects"]) for i in removed)),
                   "disappearing_classes": sorted(c for c in source_counts if not retained_counts.get(c, 0)),
                   "retained_class_counts": dict(retained_counts), "source_class_counts": dict(source_counts),
                   "lost_group_ids": np.flatnonzero(lost).tolist(), "scene_witnesses": scene_witnesses,
                   "scene_missing_ids": [self.image_ids[i] for i in scene_missing], "scene_unknown_ids": [self.image_ids[i] for i in scene_unknown],
                   "object_coverage": object_cov, "scene_coverage": scene_cov,
                   "coverage_scope": "Frozen64 ranking proxy; binary loss checks every original object directly",
                   "ready": not np.any(witnesses == -2) and not scene_unknown and not scene_missing and not (self.protected - retained)}
        return metrics, witnesses, lost

    def evaluate(self, retained, *, cancelled=None, progress=None, verify=True):
        self._validate()
        result = self._evaluate(retained, cancelled, verify, progress)[0]
        self._validate()
        return result

    def _describe_object(self, q, witness, cancelled=None):
        obj = self.flat[q]
        result = {"label_row": int(obj["label_row"]), "class_id": int(obj["class_id"]), "group_id": int(self.group_of[q]),
                  "covered": witness >= 0, "unknown": witness == -2, "witness_image_id": None, "witness_row": None, "similarity": None}
        if witness < 0:
            return result
        own = int(self.image_of[q])
        if witness == own:
            row, score = q, 1.
        else:
            positions = np.flatnonzero((self.object_rank_images[q] == witness) &
                                       (self.object_rank_scores[q] >= self.config.object_threshold))
            row = None
            if len(positions):
                candidate = int(self.object_rank_rows[q, int(positions[0])])
                score = self._object_row_scores(q, [candidate], cancelled, verify=self.objects is not None).get(candidate)
                if score is not None and score >= self.config.object_threshold:
                    row = candidate
            if row is None and self.objects is not None:
                targets = self.image_objects[witness]
                targets = targets[(self.classes[targets] == self.classes[q]) & (self._scale(q, targets) >= self.config.rarity_scale_threshold)]
                scores = self._object_row_scores(q, targets, cancelled)
                qualifying = [(target, value) for target, value in scores.items()
                              if value is not None and value >= self.config.object_threshold]
                if not qualifying:
                    raise ValueError("Saved object coverage has no qualifying native correspondence")
                row, score = min(qualifying, key=lambda item: (-item[1], item[0]))
            if row is None:
                # The image-level proof is complete; a native crop needs cache
                # replay on demand, rather than invented correspondence data.
                result.update(witness_image_id=self.image_ids[witness], correspondence_available=False)
                return result
        result.update(witness_image_id=self.image_ids[witness], witness_row=int(self.flat[row]["label_row"]), similarity=score, correspondence_available=True)
        return result

    def _impacts(self, retained, evaluation, witnesses, lost, candidate_ids=None, cancelled=None, object_weight=None):
        retained = self._indices(retained)
        removed = sorted(set(range(self.image_count)) - retained)
        selected = removed if candidate_ids is None else sorted(self._indices(candidate_ids) - retained)
        raw_gain = np.zeros(self.image_count, np.int64); group_gain = np.zeros(self.image_count, np.int64); weight_gain = np.zeros(self.image_count, np.float64)
        partial_gain = np.zeros(self.image_count, np.int64)
        for g in np.flatnonzero(lost):
            _check(cancelled)
            members = self._group_members[self._group_offsets[g]:self._group_offsets[g + 1]]
            members = members[witnesses[members] == -1]
            intersection = None
            union = np.zeros(self.image_count, bool)
            for q in members:
                proof = self._exact("object", int(q), cancelled)
                bits = np.unpackbits(proof, bitorder="little", count=self.image_count).astype(bool)
                raw_gain += bits
                union |= bits
                intersection = bits if intersection is None else intersection & bits
            if intersection is not None:
                group_gain += intersection
                weight_gain += intersection * float(self.weights[g])
                partial_gain += union & ~intersection
        # Frozen-shortlist restoration is a separate, continuous ranking hint;
        # it never contributes to either binary appearance-loss cap.
        mask = np.zeros(self.image_count, bool); mask[list(retained)] = True
        smooth = []
        for images, scores, query_weights in [(self.object_rank_images, self.object_rank_scores, self.object_weights),
                                             (self.scene_rank_images, self.scene_rank_scores, np.full(self.image_count, 1 / max(1, self.image_count)))]:
            gains = np.zeros(self.image_count, np.float64)
            for start in range(0, len(images), BLOCK):
                _check(cancelled)
                ids = images[start:start + BLOCK]; values = scores[start:start + BLOCK]
                valid = ids >= 0
                before = np.where(valid & mask[np.maximum(ids, 0)], values, 0).max(axis=1)
                delta = np.maximum(0., values - before[:, None]) * query_weights[start:start + len(ids), None]
                gains += np.bincount(ids[valid], weights=delta[valid], minlength=self.image_count)
            smooth.append(gains)
        alpha = self.config.object_weight if object_weight is None else object_weight
        combined_gain = alpha * smooth[0] + (1 - alpha) * smooth[1]
        results = []
        for i in selected:
            _check(cancelled)
            own = self.image_objects[i]
            details = [self._describe_object(int(q), int(witnesses[q]), cancelled) for q in own]
            groups = sorted(set(int(self.group_of[q]) for q in own if lost[self.group_of[q]]))
            results.append({"image_id": self.image_ids[i], "candidate_id": self.image_ids[i], "image_index": i, "scene_witness_id": evaluation["scene_witnesses"].get(self.image_ids[i], {}).get("witness_id"),
                            "objects": details, "object_witnesses": details, "uncovered_objects": [row for row in details if not row["covered"]], "lost_group_ids": groups,
                            "restored_appearance_group_count": int(group_gain[i]), "restored_object_count": int(raw_gain[i]), "partial_restored_object_count": int(raw_gain[i]), "partially_restored_group_count": int(partial_gain[i]),
                            "restored_resemblance_proxy": float(combined_gain[i]),
                            "restored_appearance_weight_fraction": float(weight_gain[i] / self.total_weight) if self.total_weight else 0.,
                            "reason": "Restoring this image recovers its directly measured appearances; partial recoveries may leave a group affected"})
        return sorted(results, key=lambda row: (-row["restored_appearance_weight_fraction"], -row["partially_restored_group_count"], -row["restored_object_count"], -row["restored_resemblance_proxy"], row["image_id"]))

    def impacts(self, retained, candidate_ids=None, *, cancelled=None, progress=None):
        self._validate()
        metrics, witnesses, lost = self._evaluate(retained, cancelled, True, progress)
        result = self._impacts(retained, metrics, witnesses, lost, candidate_ids, cancelled)
        self._validate()
        return result

    def describe(self, candidate_id, retained, *, cancelled=None):
        return self.impacts(retained, [candidate_id], cancelled=cancelled)[0]

    def plan(self, settings, pinned_ids=(), candidate_ids=None, cancelled=None, progress=None):
        self._validate()
        settings = self._settings(settings)
        n = self.image_count
        pins = self._indices(pinned_ids) | self.protected
        candidates = set(range(n)) if candidate_ids is None else self._indices(candidate_ids)
        candidates -= pins
        retained = np.ones(n, bool)
        object_witness = self.image_of.copy()
        object_dependents = [set(map(int, rows)) for rows in self.image_objects]
        scene_dependents = [set() for _ in range(n)]
        scene_witness = np.full(n, -1, np.int32)
        lost_count = np.zeros(self.group_count, np.int64)
        k = 0; weighted = 0.; removed = []
        object_buckets = [set() for _ in range(n)]
        for q, own in enumerate(self.image_of):
            object_buckets[int(own)].add(q)
        scene_buckets = [{i} for i in range(n)]

        def ranking_delta(i):
            value = 0.
            rows = np.asarray(sorted(object_buckets[i]), np.int64)
            if len(rows):
                ids = self.object_rank_images[rows]; scores = self.object_rank_scores[rows]
                mask = (ids >= 0) & retained[np.maximum(ids, 0)]
                before = np.where(mask, scores, 0).max(axis=1)
                after = np.where(mask & (ids != i), scores, 0).max(axis=1)
                value += settings["object_weight"] * float((before - after) @ self.object_weights[rows])
            rows = np.asarray(sorted(scene_buckets[i]), np.int64)
            if len(rows):
                ids = self.scene_rank_images[rows]; scores = self.scene_rank_scores[rows]
                mask = (ids >= 0) & retained[np.maximum(ids, 0)]
                before = np.where(mask, scores, 0).max(axis=1)
                after = np.where(mask & (ids != i), scores, 0).max(axis=1)
                value += (1 - settings["object_weight"]) * float((before - after).sum() / max(1, n))
            return max(0., value)

        def update_buckets(i, buckets, images, scores):
            for q in list(buckets[i]):
                ids = images[q]; valid = (ids >= 0) & retained[np.maximum(ids, 0)]
                if np.any(valid):
                    positions = np.flatnonzero(valid); best = positions[np.argmax(scores[q, positions])]
                    buckets[int(ids[best])].add(q)
            buckets[i].clear()

        initial_objects, initial_scenes = self._initial_ranking_costs()
        heap = [(max(0., settings["object_weight"] * float(initial_objects[i]) + (1 - settings["object_weight"]) * float(initial_scenes[i])), self.image_ids[i], i) for i in candidates]
        heapq.heapify(heap)
        curve = [{"removed_count": 0, "removed_percent": 0., "affected_appearance_group_count": 0, "lost_appearance_weight_fraction": 0.}]
        while heap and len(removed) < min(n, settings["max_removed_images"]):
            _check(cancelled)
            previous, _, i = heapq.heappop(heap)
            current = ranking_delta(i)
            if current > previous + 1e-12:
                heapq.heappush(heap, (current, self.image_ids[i], i))
                continue
            retained[i] = False
            # Recheck all earlier removals whose direct scene witness would
            # disappear, as well as the newly removed image. Chains never count.
            changed_scene = {}
            for q in sorted(scene_dependents[i] | {i}):
                witness = self._witness("scene", q, retained, cancelled)
                if witness < 0:
                    changed_scene = None
                    break
                changed_scene[q] = witness
            if changed_scene is None:
                retained[i] = True
                continue
            changed_objects = {}; new_groups = set(); new_weight = 0.; over_budget = False
            for q in sorted(object_dependents[i]):
                _check(cancelled)
                witness = self._witness("object", q, retained, cancelled)
                changed_objects[q] = witness
                if witness < 0:
                    group = int(self.group_of[q])
                    if not lost_count[group] and group not in new_groups:
                        new_groups.add(group)
                        new_weight = float(self.weights[sorted(new_groups)].sum())
                        fraction = (weighted + new_weight) / self.total_weight if self.total_weight else 0.
                        if k + len(new_groups) > settings["max_affected_appearance_groups"] or fraction > settings["max_lost_appearance_weight_fraction"] + 1e-12:
                            # Losses can only increase during this trial. Keep
                            # completed proofs, but do not resolve more boxes or
                            # commit any tentative witness/dependency changes.
                            over_budget = True
                            break
            if over_budget:
                retained[i] = True
                continue
            new_groups = sorted(new_groups)
            fraction = (weighted + new_weight) / self.total_weight if self.total_weight else 0.
            if k + len(new_groups) > settings["max_affected_appearance_groups"] or fraction > settings["max_lost_appearance_weight_fraction"] + 1e-12:
                retained[i] = True
                continue
            for q, witness in changed_scene.items():
                old = int(scene_witness[q])
                if old >= 0:
                    scene_dependents[old].discard(q)
                scene_witness[q] = witness
                scene_dependents[witness].add(q)
            for q, witness in changed_objects.items():
                object_witness[q] = witness
                if witness >= 0:
                    object_dependents[witness].add(q)
                else:
                    lost_count[self.group_of[q]] += 1
            object_dependents[i].clear()
            update_buckets(i, object_buckets, self.object_rank_images, self.object_rank_scores)
            update_buckets(i, scene_buckets, self.scene_rank_images, self.scene_rank_scores)
            k += len(new_groups); weighted += new_weight; removed.append(i)
            curve.append({"removed_count": len(removed), "removed_percent": 100 * len(removed) / max(1, n),
                          "affected_appearance_group_count": k, "lost_appearance_weight_fraction": weighted / self.total_weight if self.total_weight else 0.,
                          "last_image_id": self.image_ids[i], "marginal_ranking_loss": current})
            _notify(progress, len(removed), settings["max_removed_images"], "plan")
        kept = np.flatnonzero(retained).tolist()
        metrics, witnesses, lost = self._evaluate(kept, cancelled, True, progress)
        if not metrics["ready"]:
            raise RuntimeError("Final coverage plan has no direct surviving scene witness")
        if metrics["affected_appearance_group_count"] > settings["max_affected_appearance_groups"] or metrics["lost_appearance_weight_fraction"] > settings["max_lost_appearance_weight_fraction"] + 1e-12:
            raise RuntimeError("Final coverage plan exceeds the binary appearance budgets")
        impact = self._impacts(kept, metrics, witnesses, lost, cancelled=cancelled, object_weight=settings["object_weight"])
        result = {"policy": "budgeted_coverage", "settings": settings, "accounting_digest": self.accounting_digest,
                  "proof_generation": metrics["proof_generation"], "ready": True,
                  "removed_indices": metrics["removed_indices"], "retained_indices": kept,
                  "removed_ids": metrics["removed_ids"], "retained_ids": metrics["retained_ids"],
                  "removed_count": len(removed), "removed_percent": metrics["removed_percent"],
                  "removal_order": removed, "curve": curve, "metrics": metrics, "impacts": impact,
                  "pinned_ids": [self.image_ids[i] for i in sorted(pins)], "no_refill": candidate_ids is not None,
                  "ranking_scope": "Frozen64-image approximate resemblance proxy; exact expansion does not change it"}
        result["plan_digest"] = digest({key: result[key] for key in ["settings", "accounting_digest", "removed_ids", "retained_ids", "proof_generation"]})
        self._validate()
        return result


def _binding(images, config, source_digest, recipe_digest):
    return {"schema": SCHEMA, "source_digest": source_digest, "recipe_digest": recipe_digest,
            "object_threshold": config.object_threshold, "scale_threshold": config.rarity_scale_threshold,
            "scene_threshold": config.scene_threshold, "negative_scene_threshold": config.negative_scene_threshold,
            "scale_protection": config.scale_protection, "rank_k": RANK_K,
            "ordering_digest": digest([(str(im["image_id"]), str(im.get("source_group") or im.get("video_id") or im["image_id"])) for im in images]),
            "rows_digest": digest([{"id": str(im["image_id"]), "dimensions": [im["width"], im["height"]],
                "errors": im.get("protection_errors", []),
                "objects": [(o["class_id"], o["label_row"], o["feature_index"], o["box"]) for o in im["objects"]]} for im in images])}


def recompute_exact_rarity(directory, images, scene, objects, config, *, source_digest, recipe_digest,
                           progress=None, cancelled=None, neighbors_fn=None):
    """Upgrade a completed v1 accounting artifact, retaining groups/rankings.

    Caller must own this accounting directory exclusively; a new accounting
    digest deliberately invalidates old previews/approvals. No features are
    encoded, no source descriptors are changed, and cancellation before the
    short publication phase leaves the previous complete generation intact.
    """
    _check(cancelled)
    config = config if isinstance(config, PruningConfig) else PruningConfig.from_payload(config)
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    binding = _binding(images, config, source_digest, recipe_digest)
    old_binding = manifest.get("binding", {})
    if not manifest.get("complete") or manifest.get("schema") not in (SCHEMA, "pruning-direct-appearance-coverage-v1", "pruning-direct-appearance-coverage-v2") or \
       {k: v for k, v in old_binding.items() if k not in ("schema", "ordering_digest")} != {k: v for k, v in binding.items() if k not in ("schema", "ordering_digest")} or \
       ("ordering_digest" in old_binding and old_binding["ordering_digest"] != binding["ordering_digest"]):
        raise ValueError("Saved coverage belongs to another source or feature/threshold generation")
    index = CoverageIndex(directory, images, scene, objects, config, binding, cancelled)
    try:
        index._load(manifest)
        group_of = np.asarray(index.group_of).copy()
        representatives = np.asarray(index.representatives).copy()
        weights, isolation = index._rarity(representatives, cancelled, neighbors_fn, progress, group_of)
        index._validate()
        manifest = index._publish(group_of, representatives, weights, isolation, cancelled)
        index._load(manifest)
        return index
    except BaseException:
        index.close()
        raise


def prepare_coverage(directory, images, scene, objects, config, *, source_digest, recipe_digest,
                     progress=None, cancelled=None, neighbors_fn=None):
    """Prepare/reopen an immutable accounting generation using cached vectors.

    ``neighbors_fn`` can accept (class_matrix,k,source_image_indices) or the
    legacy two-argument form. It may return indices or (indices,distances,meta).
    Missing vectors on reopening permit existing proofs/read-only descriptions,
    but never trigger inference when a new direct comparison is needed.
    """
    _check(cancelled)
    config = config if isinstance(config, PruningConfig) else PruningConfig.from_payload(config)
    if config.evidence_mode != "compound" or config.scale_protection != "native_pixels_and_occupancy":
        raise ValueError("Budgeted coverage needs compound features and modern native-scale checks")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = _binding(images, config, source_digest, recipe_digest)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else None
    if manifest is not None and (manifest.get("binding") != binding or not manifest.get("complete")):
        raise ValueError("Saved coverage belongs to another source or feature/threshold generation")
    index = CoverageIndex(directory, images, scene, objects, config, binding, cancelled)
    try:
        if manifest is None:
            index._build(neighbors_fn, cancelled, progress)
            manifest = json.loads(path.read_text())
        if manifest["image_ids"] != index.image_ids or manifest["object_count"] != index.object_count:
            raise ValueError("Coverage source row mapping changed")
        index._load(manifest)
        return index
    except BaseException:
        index.close()
        raise
