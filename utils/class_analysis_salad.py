from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA = "sam3_mask_salad_fusion_v1"
CLASS_ANALYSIS_SALAD_DINO_RATIO = 0.75
CLASS_ANALYSIS_SALAD_SAM_RATIO = 0.25
CLASS_ANALYSIS_SALAD_DEFAULT_PRESET = "balanced"
CLASS_ANALYSIS_SALAD_DEFAULT_MAX_TRAIN_OBJECTS = 1024
CLASS_ANALYSIS_SALAD_DEFAULT_TOKEN_BUDGET_MB = 768
CLASS_ANALYSIS_SALAD_ESTIMATED_PATCH_TOKENS = 196
CLASS_ANALYSIS_SALAD_ESTIMATED_CHANNELS = 768

CLASS_ANALYSIS_SALAD_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "label": "Balanced",
        "num_clusters": 16,
        "cluster_dim": 32,
        "token_dim": 128,
        "hidden_dim": 256,
        "batch_size": 24,
        "weight": 0.10,
    },
    "large": {
        "label": "Large",
        "num_clusters": 64,
        "cluster_dim": 128,
        "token_dim": 256,
        "hidden_dim": 512,
        "batch_size": 8,
        "weight": 0.10,
    },
}


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(minimum, min(maximum, parsed))


def normalize_class_analysis_salad_preset(value: Any) -> str:
    preset = str(value or CLASS_ANALYSIS_SALAD_DEFAULT_PRESET).strip().lower()
    if preset in {"big", "full", "default"}:
        preset = "large"
    return preset if preset in CLASS_ANALYSIS_SALAD_PRESETS else CLASS_ANALYSIS_SALAD_DEFAULT_PRESET


@dataclass(frozen=True)
class ClassAnalysisSALADSettings:
    preset: str
    num_clusters: int
    cluster_dim: int
    token_dim: int
    hidden_dim: int
    dropout: float
    sinkhorn_iters: int
    sinkhorn_reg: float
    epochs: int
    batch_size: int
    max_train_objects: int
    token_budget_mb: int
    weight: float
    learning_rate: float
    weight_decay: float
    temperature: float

    @property
    def descriptor_dim(self) -> int:
        return self.token_dim + self.num_clusters * self.cluster_dim

    @property
    def token_budget_bytes(self) -> int:
        return self.token_budget_mb * 1024 * 1024

    @property
    def fusion_weights(self) -> Tuple[float, float, float]:
        remainder = 1.0 - self.weight
        return (
            remainder * CLASS_ANALYSIS_SALAD_DINO_RATIO,
            remainder * CLASS_ANALYSIS_SALAD_SAM_RATIO,
            self.weight,
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["descriptor_dim"] = self.descriptor_dim
        dino_weight, sam_weight, salad_weight = self.fusion_weights
        value["dino_weight"] = dino_weight
        value["sam_weight"] = sam_weight
        value["salad_weight"] = salad_weight
        return value

    def to_request_fields(self) -> Dict[str, Any]:
        return {
            "salad_preset": self.preset,
            "salad_num_clusters": self.num_clusters,
            "salad_cluster_dim": self.cluster_dim,
            "salad_token_dim": self.token_dim,
            "salad_hidden_dim": self.hidden_dim,
            "salad_dropout": self.dropout,
            "salad_sinkhorn_iters": self.sinkhorn_iters,
            "salad_sinkhorn_reg": self.sinkhorn_reg,
            "salad_epochs": self.epochs,
            "salad_batch_size": self.batch_size,
            "salad_max_train_objects": self.max_train_objects,
            "salad_token_budget_mb": self.token_budget_mb,
            "salad_weight": self.weight,
            "salad_learning_rate": self.learning_rate,
            "salad_weight_decay": self.weight_decay,
            "salad_temperature": self.temperature,
        }


def class_analysis_salad_settings(payload: Mapping[str, Any]) -> ClassAnalysisSALADSettings:
    preset = normalize_class_analysis_salad_preset(payload.get("salad_preset"))
    defaults = CLASS_ANALYSIS_SALAD_PRESETS[preset]
    return ClassAnalysisSALADSettings(
        preset=preset,
        num_clusters=_bounded_int(payload.get("salad_num_clusters"), defaults["num_clusters"], 4, 128),
        cluster_dim=_bounded_int(payload.get("salad_cluster_dim"), defaults["cluster_dim"], 8, 256),
        token_dim=_bounded_int(payload.get("salad_token_dim"), defaults["token_dim"], 8, 1024),
        hidden_dim=_bounded_int(payload.get("salad_hidden_dim"), defaults["hidden_dim"], 64, 2048),
        dropout=_bounded_float(payload.get("salad_dropout"), 0.3, 0.0, 0.8),
        sinkhorn_iters=_bounded_int(payload.get("salad_sinkhorn_iters"), 3, 1, 10),
        sinkhorn_reg=_bounded_float(payload.get("salad_sinkhorn_reg"), 1.0, 0.01, 10.0),
        epochs=_bounded_int(payload.get("salad_epochs"), 8, 1, 64),
        batch_size=_bounded_int(payload.get("salad_batch_size"), defaults["batch_size"], 2, 128),
        max_train_objects=_bounded_int(
            payload.get("salad_max_train_objects"),
            CLASS_ANALYSIS_SALAD_DEFAULT_MAX_TRAIN_OBJECTS,
            2,
            16384,
        ),
        token_budget_mb=_bounded_int(
            payload.get("salad_token_budget_mb"),
            CLASS_ANALYSIS_SALAD_DEFAULT_TOKEN_BUDGET_MB,
            64,
            8192,
        ),
        weight=_bounded_float(payload.get("salad_weight"), defaults["weight"], 0.01, 0.40),
        learning_rate=_bounded_float(payload.get("salad_learning_rate"), 1.0e-4, 1.0e-7, 0.1),
        weight_decay=_bounded_float(payload.get("salad_weight_decay"), 1.0e-4, 0.0, 1.0),
        temperature=_bounded_float(payload.get("salad_temperature"), 0.07, 0.001, 1.0),
    )


def class_analysis_salad_effective_train_limit(settings: ClassAnalysisSALADSettings) -> int:
    bytes_per_object = 2 * (
        CLASS_ANALYSIS_SALAD_ESTIMATED_PATCH_TOKENS * CLASS_ANALYSIS_SALAD_ESTIMATED_CHANNELS
        + CLASS_ANALYSIS_SALAD_ESTIMATED_CHANNELS
    ) * np.dtype(np.float16).itemsize
    budget_limit = max(2, settings.token_budget_bytes // max(1, bytes_per_object))
    return max(2, min(settings.max_train_objects, int(budget_limit)))


def _stable_digest(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def class_analysis_salad_training_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> List[int]:
    """Select a deterministic source-diverse reservoir without reading labels."""
    total = len(records)
    if total <= limit:
        return list(range(total))
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        source = str(record.get("_image_path") or record.get("image_path") or f"index:{index}")
        groups.setdefault(source, []).append(index)
    ordered_groups: List[List[int]] = []
    for source in sorted(groups, key=lambda value: _stable_digest(seed, "source", value)):
        ordered_groups.append(
            sorted(
                groups[source],
                key=lambda index: _stable_digest(
                    seed,
                    "object",
                    records[index].get("crop_cache_key")
                    or records[index].get("object_id")
                    or index,
                ),
            )
        )
    selected: List[int] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for group in ordered_groups:
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def class_analysis_salad_training_fingerprint(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    settings: ClassAnalysisSALADSettings,
    *,
    seed: int,
) -> str:
    identities = [
        str(
            records[index].get("crop_cache_key")
            or records[index].get("object_id")
            or f"{records[index].get('_image_path') or ''}:{index}"
        )
        for index in indices
    ]
    return _stable_digest(
        CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA,
        seed,
        settings.to_dict(),
        identities,
    )


def compose_class_analysis_salad_features(
    raw_features: np.ndarray,
    view_counts: Sequence[int],
    *,
    salad_dimension: int,
) -> np.ndarray:
    """Compose DINO per-view concatenation and SALAD view-mean independently."""
    matrix = np.asarray(raw_features, dtype=np.float32)
    if matrix.ndim != 2 or salad_dimension <= 0 or matrix.shape[1] <= salad_dimension:
        raise ValueError("class_analysis_salad_feature_shape_invalid")
    if sum(int(value) for value in view_counts) != matrix.shape[0]:
        raise ValueError("class_analysis_salad_view_count_mismatch")
    if len({int(value) for value in view_counts}) != 1:
        raise ValueError("class_analysis_salad_mixed_view_counts")
    dino_dimension = matrix.shape[1] - salad_dimension
    composed: List[np.ndarray] = []
    cursor = 0
    for raw_count in view_counts:
        count = int(raw_count)
        chunk = matrix[cursor : cursor + count]
        cursor += count
        dino = chunk[:, :dino_dimension]
        salad = chunk[:, dino_dimension:]
        dino /= np.maximum(np.linalg.norm(dino, axis=1, keepdims=True), 1.0e-12)
        salad /= np.maximum(np.linalg.norm(salad, axis=1, keepdims=True), 1.0e-12)
        dino_composed = dino.reshape(-1)
        dino_composed /= max(float(np.linalg.norm(dino_composed)), 1.0e-12)
        salad_composed = salad.mean(axis=0)
        salad_composed /= max(float(np.linalg.norm(salad_composed)), 1.0e-12)
        composed.append(np.concatenate([dino_composed, salad_composed]))
    return np.stack(composed, axis=0).astype(np.float32, copy=False)


def fuse_class_analysis_feature_branches(
    branches: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    if not branches or len(branches) != len(weights):
        raise ValueError("class_analysis_fusion_branch_mismatch")
    matrices = [np.asarray(branch, dtype=np.float32) for branch in branches]
    row_count = matrices[0].shape[0]
    if any(matrix.ndim != 2 or matrix.shape[0] != row_count for matrix in matrices):
        raise ValueError("class_analysis_fusion_shape_invalid")
    weight_values = np.asarray(weights, dtype=np.float64)
    if np.any(~np.isfinite(weight_values)) or np.any(weight_values < 0.0):
        raise ValueError("class_analysis_fusion_weight_invalid")
    total_weight = float(np.sum(weight_values))
    if total_weight <= 0.0:
        raise ValueError("class_analysis_fusion_weight_invalid")
    weight_values /= total_weight
    components: List[np.ndarray] = []
    for matrix, weight in zip(matrices, weight_values):
        normalized = matrix / np.maximum(
            np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-12
        )
        components.append(math.sqrt(float(weight)) * normalized)
    fused = np.concatenate(components, axis=1)
    return fused / np.maximum(
        np.linalg.norm(fused, axis=1, keepdims=True), 1.0e-12
    )
