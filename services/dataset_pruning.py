"""Image pruning from cached scene and individual, class-aware object features.

SALAD supplies scene retrieval; CSOD motivates class-balanced object coverage.
The retained order extends ingestion's k-center loop. Coverage is an explicit
descriptor proxy, never a prediction of downstream detector accuracy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import hashlib
import heapq
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from services.data_ingestion import greedy_diverse_indices_with_scores, normalize_rows

VERSION = "dataset-pruning-v3-group-excluded-audit-repair"
INDIVIDUAL_RETRIEVAL_VERSION = "source_group_exclusion_v2_audit_repair"
SCENE_THRESHOLD = 0.90
NEGATIVE_SCENE_THRESHOLD = 0.94
OBJECT_THRESHOLD = 0.90
INDEX_THRESHOLD = 0.94
INDEX_WEIGHTS = {"objects": 0.90, "scene": 0.10}
RARITY_SCALE_THRESHOLD = 0.80
NEIGHBOR_K = 32
SIMILARITY_RECIPE = {"matcher":"class_directional_appearance_v3","scene_threshold":SCENE_THRESHOLD,
                     "negative_scene_threshold":NEGATIVE_SCENE_THRESHOLD,"object_threshold":OBJECT_THRESHOLD,
                     "index_threshold":INDEX_THRESHOLD,"weights":INDEX_WEIGHTS,"neighbor_k":NEIGHBOR_K,
                     "geometry":"assignment_tie_diagnostic_only","assignment_geometry_weight":0.001,
                     "fusion":"weighted_object_scene_v2","rarity":"class_visual_scale_components_v3",
                     "rarity_scale_threshold":RARITY_SCALE_THRESHOLD,
                     "descriptor_binding":"producer_content_and_exact_geometry_v2","selection":"protected_directed_witness_prefix_v2","splits":"annotation_independent_scene_groups_v2"}


@dataclass(frozen=True)
class PruningConfig:
    scene_threshold: float = .92
    negative_scene_threshold: float = .94
    object_threshold: float = .80
    index_threshold: float = .87
    object_weight: float = .90
    rarity_scale_threshold: float = .60
    neighbor_k: int = 32
    scene_requirement: str = "required"
    evidence_mode: str = "compound"
    scale_protection: str = "native_pixels_and_occupancy"

    @classmethod
    def from_payload(cls, value=None):
        value = value or {}
        fields = cls.__dataclass_fields__
        instance = cls(**{key: value[key] for key in fields if key in value})
        for key in ["scene_threshold", "negative_scene_threshold", "object_threshold", "index_threshold", "object_weight", "rarity_scale_threshold"]:
            item = getattr(instance, key)
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or not 0 <= item <= 1:
                raise ValueError("Invalid pruning " + key)
        if type(instance.neighbor_k) is not int or not 1 <= instance.neighbor_k <= 256:
            raise ValueError("Neighbor count must be between1and256")
        if instance.scene_requirement not in {"required", "optional"} or instance.evidence_mode not in {"compound", "objects_only", "scene_only"}:
            raise ValueError("Invalid pruning evidence mode")
        if instance.scale_protection not in {"native_pixels_and_occupancy", "occupancy_legacy"}:
            raise ValueError("Invalid original scale protection")
        return instance

    @property
    def weights(self):
        weight = 1.0 if self.evidence_mode == "objects_only" else 0.0 if self.evidence_mode == "scene_only" else self.object_weight
        return {"objects": weight, "scene": 1 - weight}

    def recipe(self):
        return {**SIMILARITY_RECIPE, "config_schema": "immutable-pruning-config-v2",
                "scene_threshold": self.scene_threshold, "negative_scene_threshold": self.negative_scene_threshold,
                "object_threshold": self.object_threshold, "index_threshold": self.index_threshold,
                "weights": self.weights, "neighbor_k": self.neighbor_k, "scene_requirement": self.scene_requirement,
                "evidence_mode": self.evidence_mode, "rarity_scale_threshold": self.rarity_scale_threshold,
                "individual_object_retrieval": INDIVIDUAL_RETRIEVAL_VERSION,
                "pair_scene_prefilter": None if self.scene_requirement == 'optional' or self.evidence_mode == 'objects_only' else min(.65, self.scene_threshold, self.negative_scene_threshold),
                "branch_measurement": {'scene': self.evidence_mode != 'objects_only', 'bbox': self.evidence_mode != 'scene_only'},
                "scene_only_inventory": 'class_count_directional_v1' if self.evidence_mode == 'scene_only' else None,
                "scale_protection": self.scale_protection,
                "rarity": "class_visual_native_and_relative_scale_components_v4" if self.scale_protection == "native_pixels_and_occupancy" else SIMILARITY_RECIPE["rarity"]}


def _config(config):
    if config is None:
        # Legacy programmatic callers/archived experiment recipes keep their
        # explicit module defaults. Production jobs always pass a frozen config.
        return PruningConfig(scene_threshold=SCENE_THRESHOLD, negative_scene_threshold=NEGATIVE_SCENE_THRESHOLD,
                             object_threshold=OBJECT_THRESHOLD, index_threshold=INDEX_THRESHOLD,
                             object_weight=INDEX_WEIGHTS["objects"], rarity_scale_threshold=RARITY_SCALE_THRESHOLD, neighbor_k=NEIGHBOR_K,
                             scale_protection="occupancy_legacy")
    return config if isinstance(config, PruningConfig) else PruningConfig.from_payload(config)


def weighted_index(objects: float, scene: float, config=None) -> float:
    weights = _config(config).weights
    return weights["objects"] * objects + weights["scene"] * scene


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def geometry_key(box: list[float]) -> str:
    # Encoding uses these exact normalized coordinates, including subpixel
    # changes that can cross a floor/ceil crop boundary. Classes stay separate.
    return digest([float(v) for v in box])


def parse_objects(lines: list[str], class_count: int) -> tuple[list[dict], list[str]]:
    objects, errors = [], []
    for row, line in enumerate(lines):
        try:
            values = str(line).split()
            if len(values) != 5:
                raise ValueError("Pruning currently requires bbox labels")
            cls_value = float(values[0])
            cls = int(cls_value)
            box = [float(v) for v in values[1:]]
            if cls != cls_value or not 0 <= cls < class_count:
                raise ValueError("Unknown class ID")
            if not all(math.isfinite(v) for v in box) or not all(0 <= v <= 1 for v in box) or min(box[2:]) <= 0:
                raise ValueError("Invalid normalized bbox")
            if box[0] - box[2]/2 < -1e-5 or box[1] - box[3]/2 < -1e-5 or box[0] + box[2]/2 > 1.00001 or box[1] + box[3]/2 > 1.00001:
                raise ValueError("Bbox extends outside image")
            objects.append({"class_id": cls, "box": box, "label_row": row, "geometry_key": geometry_key(box)})
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append(f"Label row {row + 1}: {exc}")
    return objects, errors


def _geometry(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    center = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
    scale = np.abs(np.log(np.maximum(a[:, None, 2:], 1e-8) / np.maximum(b[None, :, 2:], 1e-8))).mean(axis=2)
    # Absolute center distance alone treats disjoint tiny objects as coincident.
    # Actual overlap preserves composition at the scale of each object.
    lo = np.maximum(a[:,None,:2]-a[:,None,2:]/2,b[None,:,:2]-b[None,:,2:]/2)
    hi = np.minimum(a[:,None,:2]+a[:,None,2:]/2,b[None,:,:2]+b[None,:,2:]/2)
    intersection = np.maximum(hi-lo,0).prod(axis=2)
    union = a[:,None,2:].prod(axis=2)+b[None,:,2:].prod(axis=2)-intersection
    overlap = intersection / np.maximum(union,1e-12)
    return (np.sqrt(overlap)*np.exp(-3 * center - 0.5 * scale)).astype(np.float32)


def _scale_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.abs(np.log(np.maximum(a[:,None,2:],1e-8) / np.maximum(b[None,:,2:],1e-8))).mean(axis=2)
    return np.exp(-0.5 * delta).astype(np.float32)


def native_scale_protection(boxes_left, boxes_right, dimensions_left, dimensions_right,
                            mode="native_pixels_and_occupancy") -> dict:
    """Keep native detail scale separate from appearance residualization.

    Boxes use original normalized xywh; dimensions are original image width,
    height. Encoder resize dimensions never enter this coverage protection.
    ``occupancy_legacy`` preserves the historical demo's relative-size rule.
    """
    left, right = np.asarray(boxes_left), np.asarray(boxes_right)
    occupancy = _scale_similarity(left, right)
    if mode == "occupancy_legacy":
        return {"scale": occupancy, "occupancy_scale": occupancy,
                "native_bbox_scale": None, "native_image_scale": None}
    if mode != "native_pixels_and_occupancy":
        raise ValueError("Unknown original-scale protection policy")
    dims_left = np.asarray(dimensions_left, np.float64)
    dims_right = np.asarray(dimensions_right, np.float64)
    sizes_left = np.asarray(left[:, 2:], np.float64) * dims_left
    sizes_right = np.asarray(right[:, 2:], np.float64) * dims_right
    bbox = np.exp(-.5 * np.abs(np.log(np.maximum(sizes_left[:, None], 1e-8)
                                     / np.maximum(sizes_right[None], 1e-8))).mean(axis=2))
    image = float(np.exp(-.5 * np.abs(np.log(np.maximum(dims_left, 1)
                                            / np.maximum(dims_right, 1))).mean()))
    return {"scale": np.minimum(occupancy, np.minimum(bbox, image)),
            "occupancy_scale": occupancy, "native_bbox_scale": bbox,
            "native_image_scale": image}


def pair_evidence(left: dict, right: dict, scene_similarity: float, object_features: np.ndarray, config=None) -> dict:
    """Directional visual coverage; every candidate object needs its own match."""
    config = _config(config)
    a, b = left["objects"], right["objects"]
    classes = sorted({o["class_id"] for o in a + b})
    matches, cover_a, cover_b = [], {}, {}
    minimum_a, minimum_b, minimum_geometry = 1.0, 1.0, 1.0
    unmatched_a, unmatched_b = [], []
    counts_equal = Counter(o["class_id"] for o in a) == Counter(o["class_id"] for o in b)
    exact_image = left["image_sha256"] == right["image_sha256"]
    same_exact_labels = sorted((o["class_id"],o["geometry_key"]) for o in a) == sorted((o["class_id"],o["geometry_key"]) for o in b)
    complete = not left.get("protection_errors") and not right.get("protection_errors")
    if config.evidence_mode == "scene_only":
        scene = float(np.clip(scene_similarity, 0, 1))
        compatible = complete and (not exact_image or same_exact_labels)
        counts_a, counts_b = Counter(o['class_id'] for o in a), Counter(o['class_id'] for o in b)
        inventory_a = {str(cls): min(1., counts_b[cls] / count) for cls, count in counts_a.items()}
        inventory_b = {str(cls): min(1., counts_a[cls] / count) for cls, count in counts_b.items()}
        valid_a = direction_is_redundant(config, a, b, [], scene, scene, compatible)
        valid_b = direction_is_redundant(config, b, a, [], scene, scene, compatible)
        return {"scene": scene, "object": 0., "combined": scene, "valid": valid_a or valid_b,
                "left_valid": valid_a, "right_valid": valid_b, "left_object_count": len(a), "right_object_count": len(b),
                "left_object": 0., "right_object": 0., "left_combined": scene, "right_combined": scene,
                "minimum_object_appearance": 0., "minimum_geometry": 0., "left_minimum_object_appearance": 0., "right_minimum_object_appearance": 0.,
                "left_unmatched_rows": [o["label_row"] for o in a], "right_unmatched_rows": [o["label_row"] for o in b],
                "left_matched_fraction": 0., "right_matched_fraction": 0., "counts_equal": counts_equal,
                "left_class_coverage": inventory_a, "right_class_coverage": inventory_b, "matches": [], "exact_image": exact_image,
                "object_evidence_used": False, "scene_evidence_used": True,
                "inventory_evidence_used": True, "assignment_identity": pair_assignment_identity(left, right)}
    for cls in classes:
        ai = [o for o in a if o["class_id"] == cls]
        bi = [o for o in b if o["class_id"] == cls]
        if not ai or not bi:
            if ai:
                cover_a[str(cls)] = 0.0; minimum_a = 0.0
                unmatched_a.extend(o["label_row"] for o in ai)
            if bi:
                cover_b[str(cls)] = 0.0; minimum_b = 0.0
                unmatched_b.extend(o["label_row"] for o in bi)
            continue
        fa = object_features[[o["feature_index"] for o in ai]]
        fb = object_features[[o["feature_index"] for o in bi]]
        appearance = np.clip(fa @ fb.T, 0, 1)
        geometry = _geometry(np.asarray([o["box"] for o in ai]), np.asarray([o["box"] for o in bi]))
        protection = native_scale_protection([o['box'] for o in ai], [o['box'] for o in bi],
                                             [left['width'], left['height']], [right['width'], right['height']],
                                             config.scale_protection)
        scale, occupancy_scale = protection['scale'], protection['occupancy_scale']
        native_bbox_scale, native_image_scale = protection['native_bbox_scale'], protection['native_image_scale']
        # Appearance owns matching. Original position only resolves almost-equal
        # assignments; camera displacement cannot turn a strong match into zero.
        affinity = appearance + SIMILARITY_RECIPE["assignment_geometry_weight"] * geometry
        eligible = appearance >= config.object_threshold
        # First minimize invalid assignments. A full feasible matching must
        # beat an average-score optimum containing one below-threshold object.
        penalty = min(len(ai),len(bi))+1
        rows, cols = linear_sum_assignment(-affinity + (~eligible)*penalty)
        accepted = appearance[rows,cols] >= config.object_threshold
        score = float(appearance[rows[accepted],cols[accepted]].sum())
        cover_a[str(cls)] = score / len(ai)
        cover_b[str(cls)] = score / len(bi)
        matched_a,matched_b = set(rows[accepted]),set(cols[accepted])
        unmatched_a.extend(o["label_row"] for i,o in enumerate(ai) if i not in matched_a)
        unmatched_b.extend(o["label_row"] for i,o in enumerate(bi) if i not in matched_b)
        minimum_a = min(minimum_a, float(appearance[rows,cols].min()) if len(matched_a)==len(ai) else 0.0)
        minimum_b = min(minimum_b, float(appearance[rows,cols].min()) if len(matched_b)==len(bi) else 0.0)
        minimum_geometry = min(minimum_geometry, float(geometry[rows, cols].min()))
        for i, j in zip(rows, cols):
            matches.append({"left_row": ai[i]["label_row"], "right_row": bi[j]["label_row"], "class_id": cls,
                            "appearance": float(appearance[i,j]), "geometry": float(geometry[i,j]),
                            "scale": float(scale[i,j]), "accepted": bool(appearance[i,j] >= config.object_threshold),
                            "occupancy_scale": float(occupancy_scale[i,j]),
                            "native_bbox_scale": float(native_bbox_scale[i,j]) if native_bbox_scale is not None else None,
                            "native_image_scale": native_image_scale,
                            "left_feature": ai[i]["feature_index"], "right_feature": bi[j]["feature_index"]})
    object_a = float(np.mean(list(cover_a.values()))) if cover_a else 1.0
    object_b = float(np.mean(list(cover_b.values()))) if cover_b else 1.0
    scene = float(np.clip(scene_similarity, 0, 1))
    # Explicit negatives have no object evidence. They must not receive a free
    # 90% object score, or be covered by a positively annotated image.
    combined_a = weighted_index(object_a,scene,config) if a else scene
    combined_b = weighted_index(object_b,scene,config) if b else scene
    compatible = complete and (not exact_image or same_exact_labels)
    valid_a = direction_is_redundant(config, a, b, unmatched_a, scene, combined_a, compatible)
    valid_b = direction_is_redundant(config, b, a, unmatched_b, scene, combined_b, compatible)
    return {"scene": scene, "object": max(object_a,object_b), "combined": max(combined_a,combined_b),
            "valid": valid_a or valid_b,"left_valid":valid_a,"right_valid":valid_b,
            "left_object_count":len(a),"right_object_count":len(b),
            "left_object":object_a,"right_object":object_b,"left_combined":combined_a,"right_combined":combined_b,
            "minimum_object_appearance": min(minimum_a,minimum_b), "minimum_geometry": minimum_geometry,
            "left_minimum_object_appearance":minimum_a,"right_minimum_object_appearance":minimum_b,
            "left_unmatched_rows":sorted(unmatched_a),"right_unmatched_rows":sorted(unmatched_b),
            "left_matched_fraction":1-len(unmatched_a)/len(a) if a else 1.0,
            "right_matched_fraction":1-len(unmatched_b)/len(b) if b else 1.0,
            "counts_equal": counts_equal, "left_class_coverage": cover_a, "right_class_coverage": cover_b,
            "matches": matches, "exact_image": exact_image,
            "object_evidence_used": True, "scene_evidence_used": config.evidence_mode != 'objects_only',
            "assignment_identity": pair_assignment_identity(left, right)}


def _assignment_image_input(image):
    return {'image_sha256': image['image_sha256'], 'dimensions': [image['width'], image['height']],
            'objects': [(obj['class_id'], obj['geometry_key'], obj['label_row'], obj['feature_index']) for obj in image['objects']],
            'protection_errors': image.get('protection_errors', [])}


def pair_assignment_identity(left, right):
    """Bind cached assignments to the exact directional labelled inputs."""
    return digest([_assignment_image_input(image) for image in [left, right]])


def direction_is_redundant(config, objects, other, unmatched, scene, score, compatible):
    """One gate implementation for freshly measured and cached assignments."""
    if config.evidence_mode == 'scene_only':
        counts, other_counts = Counter(obj['class_id'] for obj in objects), Counter(obj['class_id'] for obj in other)
        covered = all(other_counts[cls] >= count for cls, count in counts.items())
        floor = config.scene_threshold if objects else config.negative_scene_threshold
        return bool(compatible and bool(objects) == bool(other) and covered and scene >= max(floor, config.index_threshold))
    if not objects:
        return bool(config.evidence_mode != 'objects_only' and compatible and not other and scene >= config.negative_scene_threshold)
    return bool(compatible and not unmatched and (config.evidence_mode == 'objects_only' or config.scene_requirement == 'optional' or scene >= config.scene_threshold) and score >= config.index_threshold)


def reevaluate_pair_evidence(left, right, evidence, config, *, assignment_config, _assignment_identity=None):
    """Reuse measured assignments only for gate/weight changes.

    The caller retains the same bound feature artifact. Object threshold,
    branch mode, original-scale policy or any labelled input change requires
    measuring pair_evidence again; changing the object gate can change the
    Hungarian optimum, rather than merely its accepted flag.
    """
    config, original = _config(config), _config(assignment_config)
    if any(getattr(config, key) != getattr(original, key) for key in ['object_threshold', 'evidence_mode', 'scale_protection']):
        raise ValueError('Cached object assignments require the original object gate, branch mode and scale policy')
    expected_identity = pair_assignment_identity(left, right) if _assignment_identity is None else _assignment_identity
    if evidence.get('assignment_identity') != expected_identity:
        raise ValueError('Cached assignments do not match these exact labelled image inputs')
    result = dict(evidence)
    scene = evidence['scene']; a, b = left['objects'], right['objects']
    compatible = not left.get('protection_errors') and not right.get('protection_errors') and (left['image_sha256'] != right['image_sha256'] or
                 sorted((o['class_id'], o['geometry_key']) for o in a) == sorted((o['class_id'], o['geometry_key']) for o in b))
    scores = []
    for prefix, objects, other in [('left', a, b), ('right', b, a)]:
        score = scene if config.evidence_mode == 'scene_only' or not objects else weighted_index(evidence[prefix + '_object'], scene, config)
        result[prefix + '_combined'] = score
        result[prefix + '_valid'] = direction_is_redundant(config, objects, other, evidence[prefix + '_unmatched_rows'], scene, score, compatible)
        scores.append(score)
    result['combined'] = max(scores); result['valid'] = result['left_valid'] or result['right_valid']
    return result


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        x, y = self.find(a), self.find(b)
        if x != y:
            self.parent[max(x, y)] = min(x, y)


def build_evidence(images: list[dict], scene_features: np.ndarray, object_features: np.ndarray,
                   neighbors_fn: Callable, progress: Callable = lambda *a: None,
                   cancelled: Callable = lambda: False, store=None, config=None, retrieval_audit=None, group_neighbors_fn=None):
    config = _config(config)
    n = len(images)
    scene = normalize_rows(scene_features)
    scene_prefilter = config.recipe()['pair_scene_prefilter']
    # Streamed component rows already have independent unit normalization.
    objects = object_features
    from services.dataset_pruning_evidence import CandidatePairs
    candidates = CandidatePairs(Path(store.path).parent if store is not None else Path(tempfile.gettempdir()))
    evidence = store if store is not None else []
    def retrieve(matrix, route, *, class_id=None, image_indices=None):
        excluded = image_indices is not None and group_neighbors_fn is not None
        neighbors = group_neighbors_fn(matrix, config.neighbor_k, image_indices) if excluded else neighbors_fn(matrix, config.neighbor_k)
        record = {'route': route, 'query_count': len(matrix), 'neighbor_k': config.neighbor_k,
                  'budget_scope': ('cross-image object neighbors after source-image exclusion' if excluded else 'unfiltered object neighbors before same-image exclusion') if image_indices is not None else 'image neighbors',
                  'version': INDIVIDUAL_RETRIEVAL_VERSION if excluded else 'unfiltered_neighbors_v2',
                  'class_id': class_id}
        if image_indices is not None:
            same, cross = np.zeros(len(matrix), np.int64), np.zeros(len(matrix), np.int64)
            for query, row in enumerate(neighbors):
                for neighbor in row:
                    if 0 <= int(neighbor) < len(image_indices):
                        if image_indices[int(neighbor)] == image_indices[query]: same[query] += 1
                        else: cross[query] += 1
            record.update(same_image_neighbors=int(same.sum()), cross_image_neighbors=int(cross.sum()),
                          zero_cross_image_queries=int(np.count_nonzero(cross == 0)),
                          mean_cross_image_neighbors=float(cross.mean()) if len(cross) else 0.,
                          cross_image_neighbor_quantiles=np.quantile(cross, [0, .25, .5, .75, 1]).tolist() if len(cross) else [],
                          recall_scope='Shared seeded audit measures group-excluded neighbors' if excluded else 'Shared ANN audit measures unfiltered neighbors; effective cross-image recall is not measured by that audit')
        if retrieval_audit is not None: retrieval_audit(record)
        return neighbors

    try:
        if config.evidence_mode != "objects_only":
            neighbors = retrieve(scene, 'whole_image')
            candidates.update((min(i,int(j)), max(i,int(j))) for i, row in enumerate(neighbors) for j in row if 0 <= j < n and i != j)
        # Object retrieval complements scene retrieval; final matching remains
        # class-aware and individual. No dense all-pairs matrix is retained.
        if len(objects) and config.evidence_mode != "scene_only":
            prototypes = np.zeros((n, objects.shape[1]), np.float32)
            for i, image in enumerate(images):
                ids = [o["feature_index"] for o in image["objects"]]
                if ids: prototypes[i] = objects[ids].mean(axis=0)
            for i, row in enumerate(retrieve(normalize_rows(prototypes), 'mean_object_prototype')):
                candidates.update((min(i,int(j)), max(i,int(j))) for j in row if 0 <= j < n and i != j)
            by_class = defaultdict(list)
            for image_index, image in enumerate(images):
                for obj in image["objects"]: by_class[obj["class_id"]].append((image_index, obj["feature_index"]))
            for class_id, entries in by_class.items():
                if cancelled(): raise RuntimeError("cancelled")
                positions = np.asarray([feature for _, feature in entries], dtype=np.int64)
                # Reuse the DQE ANN with a spillable class representation.
                if len(positions)*objects.shape[1]*4 > 64*1024*1024:
                    temporary = tempfile.NamedTemporaryFile(prefix="pruning-class-ann-", suffix=".npy", delete=False)
                    temporary.close()
                    class_matrix = np.lib.format.open_memmap(temporary.name, mode="w+", dtype=np.float32, shape=(len(positions), objects.shape[1]))
                    try:
                        for start in range(0, len(positions), 512): class_matrix[start:start+512] = objects[positions[start:start+512]]
                        class_matrix.flush()
                        feature_neighbors = retrieve(class_matrix, 'individual_objects', class_id=class_id, image_indices=[entry[0] for entry in entries])
                    finally:
                        class_matrix._mmap.close()
                        Path(temporary.name).unlink(missing_ok=True)
                else: feature_neighbors = retrieve(objects[positions], 'individual_objects', class_id=class_id, image_indices=[entry[0] for entry in entries])
                for ordinal, row in enumerate(feature_neighbors):
                    image_index = entries[ordinal][0]
                    for other in row:
                        if 0 <= int(other) < len(entries):
                            partner = entries[int(other)][0]
                            if partner != image_index: candidates.add((min(image_index, partner), max(image_index, partner)))
        exact = defaultdict(list)
        for i, image in enumerate(images): exact[image["image_sha256"]].append(i)
        # A star groups exact duplicates beyond retrieval k. Label conflicts
        # still block directional pruning.
        for ids in exact.values(): candidates.update((ids[0], i) for i in ids[1:])
        total = len(candidates)
        for done, (a,b) in enumerate(candidates, 1):
            if cancelled(): raise RuntimeError("cancelled")
            similarity = float(scene[a] @ scene[b]) if config.evidence_mode != 'objects_only' else 0.
            if scene_prefilter is None or similarity >= scene_prefilter or images[a]["image_sha256"] == images[b]["image_sha256"]:
                evidence.append({"left": a, "right": b, **pair_evidence(images[a], images[b], similarity, objects, config)})
            if done % 256 == 0 or done == total: progress(done, total)
        if store is not None: store.finish()
        return evidence
    except Exception:
        if store is not None: store.abort()
        raise
    finally:
        # Retrieval, cancellation and matching failures all release the pool.
        candidates.close()


def build_order(images: list[dict], scene_features: np.ndarray, evidence: list[dict],
                keep_ids: list[str] | None = None, cancelled: Callable = lambda: False, config=None) -> dict:
    """Safe endpoint, rare exemplars and the shared ingestion retained order."""
    legacy_config = config is None
    config = _config(config)
    n = len(images)
    id_index = {image["image_id"]: i for i,image in enumerate(images)}
    unknown = set(keep_ids or []) - set(id_index)
    if unknown:
        raise ValueError("Unknown Keep override")
    adjacency, valid = [[] for _ in images], [set() for _ in images]
    groups = UnionFind(n)
    feature_count = sum(len(i["objects"]) for i in images)
    visual = UnionFind(feature_count)
    reasons = defaultdict(set)
    for i,image in enumerate(images):
        if image.get("protection_errors"):
            reasons[i].add("Incomplete or conflicting annotation evidence")
    for edge in evidence:
        a,b = edge["left"],edge["right"]
        # Keep only compact coverage updates in memory. Individual assignments
        # and appearance measurements remain in the streaming evidence store.
        adjacency[a].append((b,edge["scene"],edge["right_combined"],edge["right_class_coverage"]))
        adjacency[b].append((a,edge["scene"],edge["left_combined"],edge["left_class_coverage"]))
        # valid[witness] contains candidates covered by that retained image.
        if edge["left_valid"]: valid[b].add(a)
        if edge["right_valid"]: valid[a].add(b)
        # Split leakage is a pixel/scene relationship. Annotation disagreement
        # must protect removals without placing the same near scene in both sets.
        if (edge["scene"] >= config.scene_threshold or edge["exact_image"]
                or (config.evidence_mode == 'objects_only' and edge['valid'])):
            groups.union(a,b)
        for match in edge["matches"]:
            if match["accepted"] and match["scale"] >= config.rarity_scale_threshold:
                visual.union(match["left_feature"],match["right_feature"])
        if edge["exact_image"]:
            groups.union(a,b)
            if not edge["valid"]:
                reasons[a].add("Identical image with unresolved object or label differences")
                reasons[b].add("Identical image with unresolved object or label differences")
    class_counts = Counter(o["class_id"] for im in images for o in im["objects"])
    neighborhoods = defaultdict(list)
    for i,image in enumerate(images):
        for obj in image["objects"]:
            neighborhoods[(obj["class_id"],visual.find(obj["feature_index"]))].append(i)
    for (cls,_), ids in neighborhoods.items():
        if config.evidence_mode != "scene_only" and len(ids) <= max(3, int(math.ceil(class_counts[cls] * 0.002))):
            pinned_replacements = set(ids) & {id_index[identifier] for identifier in keep_ids or []}
            representative = min(pinned_replacements or set(ids), key=lambda i: (-len(valid[i]), images[i]["image_id"]))
            reasons[representative].add("Rare object appearance, scale or composition")
    for i in sorted({id_index[id] for id in keep_ids or []}):
        reasons[i].add("Kept by you")
    protected = sorted(reasons, key=lambda i: images[i]["image_id"])
    covered = set(protected)
    for i in protected:
        covered.update(valid[i])
    # Direct-witness dominating representatives prevent transitive removals.
    # Choosing by uncovered neighborhood size avoids preserving every member
    # of a common duplicate family. Negatives use their scene neighborhoods.
    remaining = set(range(n)) - covered
    heap = [(-len(valid[i] & remaining), images[i]["image_id"], i) for i in remaining]
    heapq.heapify(heap)
    while remaining:
        if cancelled():
            raise RuntimeError("cancelled")
        degree,_,i = heapq.heappop(heap)
        if i not in remaining:
            continue
        current_degree = -len(valid[i] & remaining)
        if degree != current_degree:
            heapq.heappush(heap,(current_degree,images[i]["image_id"],i))
            continue
        reasons[i].add("Representative of a distinct negative scene" if not images[i]["objects"] else "Representative of distinct scene and objects")
        protected.append(i)
        covered.add(i); covered.update(valid[i])
        remaining.difference_update({i} | valid[i])
    # Class coverage uses the fixed source inventory, including every instance.
    classes = sorted(class_counts)
    class_col = {cls:c for c,cls in enumerate(classes)}
    counts = np.zeros((n,len(classes)), np.float64)
    for i,image in enumerate(images):
        for obj in image["objects"]:
            counts[i,class_col[obj["class_id"]]] += 1
    best_classes = np.zeros_like(counts)
    class_totals = np.maximum(counts.sum(axis=0),1)
    best_scene = np.zeros(n,np.float32)
    best_combined = np.zeros(n,np.float32)
    scene_sum = 0.0
    class_sums = np.zeros(len(classes),np.float64)
    curve = []

    def update(idx):
        nonlocal scene_sum
        if cancelled():
            raise RuntimeError("cancelled")
        entries = [(idx,1.0,1.0,None)] + adjacency[idx]
        distances = np.ones(n,np.float32)
        for other,sc,co,cv in entries:
            scene_sum += max(0.0, sc - float(best_scene[other]))
            best_scene[other] = max(sc,best_scene[other])
            best_combined[other] = max(co,best_combined[other])
            distances[other] = 1 - best_combined[other]
            cv = {str(c):1.0 for c in classes} if cv is None else cv
            for c,value in cv.items():
                col = class_col[int(c)]
                delta = max(0.0, float(value) - best_classes[other,col])
                class_sums[col] += delta * counts[other,col]
                best_classes[other,col] = max(value,best_classes[other,col])
        return distances

    def selected(idx,count,scores):
        if count < len(protected):
            return
        per_class = class_sums / class_totals if classes else np.ones(1)
        scene_cov = min(1.0,scene_sum/n)
        object_cov = float(per_class.mean())
        curve.append({"removed_count": n-count, "removed_percent": (n-count)*100/n,
                      "scene_coverage": None if config.evidence_mode == 'objects_only' else scene_cov,
                      "object_coverage": None if config.evidence_mode == 'scene_only' else object_cov,
                      "inventory_coverage": object_cov if config.evidence_mode == 'scene_only' else None,
                      "worst_class_coverage": float(per_class.min()),
                      "coverage": weighted_index(object_cov,scene_cov,config) if classes else scene_cov})

    order,_,_ = greedy_diverse_indices_with_scores(scene_features, keep_fraction=1,
                 initial_indices=protected, distance_update_fn=update, on_select=selected)
    return {"version": VERSION, "similarity_recipe":SIMILARITY_RECIPE if legacy_config else config.recipe(),"order": order, "protected_count": len(protected),
            "max_removed_count": n-len(protected), "curve": sorted(curve,key=lambda p:p["removed_count"]),
            "protection_reasons": {str(i):sorted(r) for i,r in reasons.items()},
            "groups": [groups.find(i) for i in range(n)], "keep_ids": sorted(keep_ids or []),
            "coverage_note": "Estimated descriptor coverage of the complete source; sparse retrieval gives a lower bound. It is not measured detector accuracy.",
            "index_weights":config.weights,
            "thresholds": {"scene":config.scene_threshold,"negative_scene":config.negative_scene_threshold,"object_appearance":config.object_threshold,
                           "redundancy_index":config.index_threshold,"geometry":None,"rarity_scale":config.rarity_scale_threshold,"neighbor_k":config.neighbor_k}}


def selection(images: list[dict], result: dict, removed_count: int) -> dict:
    if isinstance(removed_count,bool) or not isinstance(removed_count,int) or not 0 <= removed_count <= result["max_removed_count"]:
        raise ValueError("Removal count exceeds the feasible protected endpoint")
    count = len(images)-removed_count
    retained = set(result["order"][:count])
    return {"retained": sorted(retained), "removed": [i for i in range(len(images)) if i not in retained],
            **result["curve"][removed_count]}


def removal_evidence(index: int, retained: set[int], evidence: list[dict]) -> dict:
    matches = []
    for edge in evidence.edges_for(index) if hasattr(evidence,"edges_for") else evidence:
        if not edge["valid"]:
            continue
        if edge["left"] == index and edge["right"] in retained and edge["left_valid"]:
            matches.append((edge["right"],False,edge))
        elif edge["right"] == index and edge["left"] in retained and edge["right_valid"]:
            matches.append((edge["left"],True,edge))
    if not matches:
        raise ValueError("Removal has no direct retained witness")
    witness,reverse,edge = max(matches,key=lambda row:(row[2]["right_combined" if row[1] else "left_combined"],-row[0]))
    side = "right" if reverse else "left"
    # Public review reports the proposed removal's direction, not the best or
    # weakest score across both inventories.
    return {**edge,"retained_index":witness,"reverse":reverse,"valid":True,
            "scene": edge['scene'] if edge.get('scene_evidence_used', True) else None,
            "object":edge[f"{side}_object"] if edge.get('object_evidence_used', True) else None,"combined":edge[f"{side}_combined"],
            "minimum_object_appearance":edge[f"{side}_minimum_object_appearance"] if edge.get('object_evidence_used', True) else None,
            "candidate_matched_fraction":edge[f"{side}_matched_fraction"],"unmatched_candidate_rows":edge[f"{side}_unmatched_rows"],
            "retained_extra_objects":edge["left_object_count" if reverse else "right_object_count"]-edge["right_object_count" if reverse else "left_object_count"]}


def grouped_splits(images: list[dict], retained: list[int], groups: list[int]) -> dict:
    """Deterministic multilabel group allocation, with honest ratio constraints."""
    source_val = sum(im.get("split") == "val" for im in images)
    ratio = source_val/len(images) if 0 < source_val < len(images) else 0.2
    buckets = defaultdict(list)
    for i in retained:
        buckets[groups[i]].append(i)
    classes = sorted({o["class_id"] for i in retained for o in images[i]["objects"]})
    targets = Counter(o["class_id"] for i in retained for o in images[i]["objects"])
    val_counts = Counter()
    train_counts = Counter()
    val, train = [], []
    ordered = sorted(buckets.values(), key=lambda ids:(-len(ids), min(images[i]["image_id"] for i in ids)))
    remaining_counts = targets.copy()
    for pos,ids in enumerate(ordered):
        counts = Counter(o["class_id"] for i in ids for o in images[i]["objects"])
        remaining_counts.subtract(counts)
        required_for_train = any(train_counts[c]==0 and remaining_counts[c]==0 for c in counts)
        def cost(add):
            size = ((len(val)+(len(ids) if add else 0))-len(retained)*ratio)/max(1,len(retained))
            label = sum(((val_counts[c]+(counts[c] if add else 0)-targets[c]*ratio)/max(1,targets[c]))**2 for c in classes)/max(1,len(classes))
            return size*size + label
        to_val = cost(True) < cost(False)
        if pos == len(ordered)-1 and len(ordered)>1:
            if not train: to_val = False
            elif not val: to_val = True
        if required_for_train: to_val = False
        (val if to_val else train).extend(ids)
        if to_val: val_counts.update(counts)
        else: train_counts.update(counts)
    warnings = []
    if not val or not train:
        warnings.append("Independent train and validation sets are infeasible with the retained duplicate groups.")
    missing = [c for c in classes if val_counts[c]==0 or val_counts[c]==targets[c]]
    if missing:
        warnings.append("Some classes are present in only one split after grouped allocation.")
    val_set = set(val)
    return {"assignments": {str(i):"val" if i in val_set else "train" for i in retained},
            "target_val_fraction": ratio, "actual_val_fraction":len(val)/max(1,len(retained)),
            "train_count":len(train),"val_count":len(val),"group_count":len(buckets),
            "class_counts":{str(c):{"train":targets[c]-val_counts[c],"val":val_counts[c]} for c in classes},
            "warnings":warnings,"method":"deterministic_multilabel_scene_groups_v2"}


def coverage_for_retained(images, retained, evidence, config=None):
    """Evaluate the actual approved variant, including unreviewed keepers."""
    config = _config(config)
    retained = set(retained)
    classes = sorted({obj['class_id'] for image in images for obj in image['objects']})
    columns = {cls: index for index, cls in enumerate(classes)}
    counts = np.zeros((len(images), len(classes)), np.float64)
    for index, image in enumerate(images):
        for obj in image['objects']: counts[index, columns[obj['class_id']]] += 1
    best_scene = np.zeros(len(images), np.float32)
    best_classes = np.zeros_like(counts)
    for index in retained:
        best_scene[index] = 1
        best_classes[index] = 1
    for edge in evidence:
        for candidate, witness, side in [(edge['left'], edge['right'], 'left'), (edge['right'], edge['left'], 'right')]:
            if witness not in retained: continue
            best_scene[candidate] = max(best_scene[candidate], edge['scene'])
            for cls, value in edge[side + '_class_coverage'].items():
                best_classes[candidate, columns[int(cls)]] = max(best_classes[candidate, columns[int(cls)]], value)
    per_class = (best_classes * counts).sum(axis=0) / np.maximum(counts.sum(axis=0), 1) if classes else np.ones(1)
    scene_coverage = float(best_scene.mean())
    object_coverage = float(per_class.mean())
    return {'removed_count': len(images)-len(retained), 'removed_percent': 100*(len(images)-len(retained))/max(1,len(images)),
            'scene_coverage': None if config.evidence_mode == 'objects_only' else scene_coverage,
            'object_coverage': None if config.evidence_mode == 'scene_only' else object_coverage,
            'inventory_coverage': object_coverage if config.evidence_mode == 'scene_only' else None,
            'worst_class_coverage': float(per_class.min()),
            'coverage': weighted_index(object_coverage, scene_coverage, config) if classes else scene_coverage}
