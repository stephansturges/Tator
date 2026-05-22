"""MLX runtime for Tator-local SALAD heads.

This is a Tator-owned MLX port of ``utils.local_salad.LocalSALADHead``.  It
keeps the same architecture and saved ``.pt`` payload contract so existing
locally trained heads can run through either Torch or MLX.  No upstream SALAD
checkpoint is loaded here.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils.local_salad import (
    LOCAL_SALAD_CACHE_VERSION,
    LOCAL_SALAD_POLICY,
    LOCAL_SALAD_TRAINER,
    LocalSALADConfig,
)

try:  # MLX is optional outside the macOS inference environment.
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as mlx_optim
except Exception:  # noqa: BLE001
    mx = None
    mlx_nn = None
    mlx_optim = None


if mlx_nn is not None:
    _MLXModuleBase = mlx_nn.Module
else:

    class _MLXModuleBase:  # type: ignore[no-redef]
        pass


def local_salad_mlx_available() -> bool:
    return mx is not None and mlx_nn is not None and mlx_optim is not None


def resolve_local_salad_backend(requested: Optional[str] = None) -> str:
    """Resolve the local SALAD head backend.

    ``auto`` is the app default.  On macOS it prefers MLX when available; all
    other cases use the existing Torch implementation unless MLX is explicitly
    requested.
    """

    raw = str(requested or os.environ.get("LOCAL_SALAD_BACKEND") or "auto").strip().lower()
    if raw in {"torch", "pytorch", "pt"}:
        return "torch"
    if raw in {"mlx", "apple_mlx"}:
        if not local_salad_mlx_available():
            raise RuntimeError("local_salad_mlx_unavailable")
        return "mlx"
    if platform.system().lower() == "darwin" and local_salad_mlx_available():
        return "mlx"
    return "torch"


def _to_mx_array(value: Any) -> Any:
    if mx is None:
        raise RuntimeError("local_salad_mlx_unavailable")
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return mx.array(np.asarray(value, dtype=np.float32))


def _mx_to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _mx_l2_normalize(value: Any, axis: int = -1) -> Any:
    denom = mx.linalg.norm(value, axis=axis, keepdims=True)
    return value / mx.maximum(denom, mx.array(1e-12, dtype=value.dtype))


def _mx_log_optimal_transport(scores: Any, dustbin_score: Any, *, num_iters: int, reg: float) -> Any:
    if scores.ndim != 3:
        raise ValueError("scores must be [batch, tokens, clusters]")
    batch, token_count, cluster_count = scores.shape
    if token_count <= 0 or cluster_count <= 0:
        raise ValueError("scores must contain at least one token and cluster")
    work = mx.transpose(scores.astype(mx.float32), (0, 2, 1)) / max(1e-6, float(reg))
    dust = mx.broadcast_to(dustbin_score.astype(mx.float32).reshape(1, 1, 1), (batch, 1, token_count))
    work = mx.concatenate([work, dust], axis=1)
    rows = cluster_count + 1
    cols = token_count
    norm = -mx.log(mx.array(float(rows + cols), dtype=mx.float32))
    log_a_main = mx.full((batch, cluster_count), norm, dtype=mx.float32)
    log_a_dust = mx.full(
        (batch, 1),
        norm + mx.log(mx.array(max(1.0, float(cols - cluster_count)), dtype=mx.float32)),
        dtype=mx.float32,
    )
    log_a = mx.concatenate([log_a_main, log_a_dust], axis=1)
    log_b = mx.full((batch, cols), norm, dtype=mx.float32)
    u = mx.zeros_like(log_a)
    v = mx.zeros_like(log_b)
    for _ in range(max(1, int(num_iters))):
        u = log_a - mx.logsumexp(work + mx.expand_dims(v, axis=1), axis=2)
        v = log_b - mx.logsumexp(work + mx.expand_dims(u, axis=2), axis=1)
    return work + mx.expand_dims(u, axis=2) + mx.expand_dims(v, axis=1) - norm


class MLXLocalSALADHead(_MLXModuleBase):
    """MLX implementation matching ``LocalSALADHead`` exactly."""

    def __init__(self, config: LocalSALADConfig):
        if not local_salad_mlx_available():
            raise RuntimeError("local_salad_mlx_unavailable")
        super().__init__()
        if int(config.num_channels) <= 0:
            raise ValueError("num_channels must be positive")
        self.config = config
        self.token_fc1 = mlx_nn.Linear(config.num_channels, config.hidden_dim)
        self.token_fc2 = mlx_nn.Linear(config.hidden_dim, config.token_dim)
        self.cluster_fc1 = mlx_nn.Linear(config.num_channels, config.hidden_dim)
        self.cluster_dropout = mlx_nn.Dropout(config.dropout) if config.dropout > 0 else mlx_nn.Identity()
        self.cluster_fc2 = mlx_nn.Linear(config.hidden_dim, config.cluster_dim)
        self.score_fc1 = mlx_nn.Linear(config.num_channels, config.hidden_dim)
        self.score_dropout = mlx_nn.Dropout(config.dropout) if config.dropout > 0 else mlx_nn.Identity()
        self.score_fc2 = mlx_nn.Linear(config.hidden_dim, config.num_clusters)
        self.dust_bin = mx.array(1.0, dtype=mx.float32)

    @property
    def output_dim(self) -> int:
        return int(self.config.token_dim + self.config.num_clusters * self.config.cluster_dim)

    def __call__(self, patch_tokens: Any, global_token: Optional[Any] = None) -> Any:
        patches = _to_mx_array(patch_tokens)
        if patches.ndim != 3:
            raise ValueError("patch_tokens must be [batch, tokens, channels]")
        if patches.shape[-1] != self.config.num_channels:
            raise ValueError(
                f"patch token width {patches.shape[-1]} does not match head width {self.config.num_channels}"
            )
        patches = mx.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
        if global_token is None:
            global_work = mx.mean(patches, axis=1)
        else:
            global_work = mx.nan_to_num(_to_mx_array(global_token), nan=0.0, posinf=0.0, neginf=0.0)
            if global_work.ndim == 3:
                global_work = global_work[:, 0, :]
            if global_work.ndim != 2 or global_work.shape[-1] != self.config.num_channels:
                global_work = mx.mean(patches, axis=1)

        token_hidden = mlx_nn.relu(self.token_fc1(global_work))
        global_desc = _mx_l2_normalize(self.token_fc2(token_hidden), axis=-1)

        cluster_hidden = mlx_nn.relu(self.cluster_dropout(self.cluster_fc1(patches)))
        local_features = mx.transpose(self.cluster_fc2(cluster_hidden), (0, 2, 1))
        score_hidden = mlx_nn.relu(self.score_dropout(self.score_fc1(patches)))
        scores = self.score_fc2(score_hidden)
        log_assign = _mx_log_optimal_transport(
            scores,
            self.dust_bin,
            num_iters=self.config.sinkhorn_iters,
            reg=self.config.sinkhorn_reg,
        )
        assign = mx.exp(log_assign[:, :-1, :])
        pooled = mx.einsum("bdn,bkn->bdk", local_features, assign)
        pooled = _mx_l2_normalize(pooled, axis=1).reshape(patches.shape[0], -1)
        return _mx_l2_normalize(mx.concatenate([global_desc, pooled], axis=-1), axis=-1)

    def load_torch_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.update(
            {
                "token_fc1": {
                    "weight": _to_mx_array(state_dict["token_features.0.weight"]),
                    "bias": _to_mx_array(state_dict["token_features.0.bias"]),
                },
                "token_fc2": {
                    "weight": _to_mx_array(state_dict["token_features.2.weight"]),
                    "bias": _to_mx_array(state_dict["token_features.2.bias"]),
                },
                "cluster_fc1": {
                    "weight": _to_mx_array(state_dict["cluster_features.0.weight"]),
                    "bias": _to_mx_array(state_dict["cluster_features.0.bias"]),
                },
                "cluster_fc2": {
                    "weight": _to_mx_array(state_dict["cluster_features.3.weight"]),
                    "bias": _to_mx_array(state_dict["cluster_features.3.bias"]),
                },
                "score_fc1": {
                    "weight": _to_mx_array(state_dict["score.0.weight"]),
                    "bias": _to_mx_array(state_dict["score.0.bias"]),
                },
                "score_fc2": {
                    "weight": _to_mx_array(state_dict["score.3.weight"]),
                    "bias": _to_mx_array(state_dict["score.3.bias"]),
                },
                "dust_bin": _to_mx_array(state_dict["dust_bin"]).reshape(()),
            }
        )
        mx.eval(self.parameters())


def is_mlx_local_salad_head(head: Any) -> bool:
    return isinstance(head, MLXLocalSALADHead)


def mlx_local_salad_state_dict(head: MLXLocalSALADHead) -> Dict[str, torch.Tensor]:
    def tensor(value: Any) -> torch.Tensor:
        return torch.from_numpy(_mx_to_numpy(value).copy())

    return {
        "dust_bin": tensor(head.dust_bin).reshape(()),
        "token_features.0.weight": tensor(head.token_fc1.weight),
        "token_features.0.bias": tensor(head.token_fc1.bias),
        "token_features.2.weight": tensor(head.token_fc2.weight),
        "token_features.2.bias": tensor(head.token_fc2.bias),
        "cluster_features.0.weight": tensor(head.cluster_fc1.weight),
        "cluster_features.0.bias": tensor(head.cluster_fc1.bias),
        "cluster_features.3.weight": tensor(head.cluster_fc2.weight),
        "cluster_features.3.bias": tensor(head.cluster_fc2.bias),
        "score.0.weight": tensor(head.score_fc1.weight),
        "score.0.bias": tensor(head.score_fc1.bias),
        "score.3.weight": tensor(head.score_fc2.weight),
        "score.3.bias": tensor(head.score_fc2.bias),
    }


def load_mlx_local_salad_head_file(path: Path) -> Tuple[MLXLocalSALADHead, Dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"local SALAD head not found: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("local_salad_head_unsupported_format")
    if payload.get("format") != LOCAL_SALAD_CACHE_VERSION:
        raise ValueError("local_salad_head_unsupported_format")
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("policy") != LOCAL_SALAD_POLICY:
        raise ValueError("local_salad_head_policy_required")
    if metadata.get("trainer") != LOCAL_SALAD_TRAINER:
        raise ValueError("local_salad_head_trainer_required")
    config = LocalSALADConfig.from_dict(payload.get("config") or {})
    head = MLXLocalSALADHead(config)
    head.load_torch_state_dict(payload.get("state_dict") or {})
    head.eval()
    return head, {"path": str(resolved), **metadata, "config": config.to_dict(), "runtime_backend": "mlx"}


def encode_local_salad_mlx(
    head: MLXLocalSALADHead,
    patch_tokens: Any,
    *,
    global_token: Optional[Any] = None,
) -> np.ndarray:
    head.eval()
    output = head(patch_tokens, global_token=global_token)
    mx.eval(output)
    return _mx_to_numpy(output)


def mlx_symmetric_infonce_loss(left: Any, right: Any, *, temperature: float = 0.07) -> Any:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left and right descriptors must have matching [batch, dim] shapes")
    if left.shape[0] <= 1:
        raise ValueError("InfoNCE needs at least two paired samples")
    temp = max(1e-4, float(temperature))
    logits = _mx_l2_normalize(left, axis=-1) @ mx.transpose(_mx_l2_normalize(right, axis=-1)) / temp
    labels = mx.arange(left.shape[0])
    return 0.5 * (
        mlx_nn.losses.cross_entropy(logits, labels, reduction="mean")
        + mlx_nn.losses.cross_entropy(mx.transpose(logits), labels, reduction="mean")
    )


def make_mlx_local_salad_optimizer(*, learning_rate: float, weight_decay: float) -> Any:
    if not local_salad_mlx_available():
        raise RuntimeError("local_salad_mlx_unavailable")
    return mlx_optim.AdamW(learning_rate=float(learning_rate), weight_decay=float(weight_decay))


def mlx_local_salad_train_step(
    head: MLXLocalSALADHead,
    optimizer: Any,
    patches_a: Any,
    global_a: Any,
    patches_b: Any,
    global_b: Any,
    *,
    temperature: float = 0.07,
    max_grad_norm: float = 1.0,
) -> float:
    pa = _to_mx_array(patches_a)
    ga = _to_mx_array(global_a)
    pb = _to_mx_array(patches_b)
    gb = _to_mx_array(global_b)
    head.train()

    def loss_fn(model: MLXLocalSALADHead, left_patches: Any, left_global: Any, right_patches: Any, right_global: Any) -> Any:
        left_desc = model(left_patches, global_token=left_global)
        right_desc = model(right_patches, global_token=right_global)
        return mlx_symmetric_infonce_loss(left_desc, right_desc, temperature=temperature)

    loss, grads = mlx_nn.value_and_grad(head, loss_fn)(head, pa, ga, pb, gb)
    if max_grad_norm and max_grad_norm > 0:
        grads, _grad_norm = mlx_optim.clip_grad_norm(grads, max_grad_norm)
    optimizer.update(head, grads)
    mx.eval(head.parameters(), optimizer.state)
    return float(np.asarray(loss))
