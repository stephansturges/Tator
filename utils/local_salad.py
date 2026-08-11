"""Locally trainable SALAD optimal-transport aggregation heads.

This module implements the SALAD aggregation mechanism we need without loading
any third-party SALAD checkpoint. The head is initialized locally and trained
on local image/video data; upstream SALAD is only an architecture reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOCAL_SALAD_CACHE_VERSION = "local-salad-v1"
LOCAL_SALAD_POLICY = "local_training_only_no_external_salad_checkpoint"
LOCAL_SALAD_TRAINER = "tator_local_salad_trainer"


@dataclass(frozen=True)
class LocalSALADConfig:
    num_channels: int
    num_clusters: int = 64
    cluster_dim: int = 128
    token_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3
    sinkhorn_iters: int = 3
    sinkhorn_reg: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalSALADConfig":
        payload = dict(data or {})
        return cls(
            num_channels=int(payload.get("num_channels") or 0),
            num_clusters=int(payload.get("num_clusters") or 64),
            cluster_dim=int(payload.get("cluster_dim") or 128),
            token_dim=int(payload.get("token_dim") or 256),
            hidden_dim=int(payload.get("hidden_dim") or 512),
            dropout=float(payload.get("dropout") if payload.get("dropout") is not None else 0.3),
            sinkhorn_iters=int(payload.get("sinkhorn_iters") or 3),
            sinkhorn_reg=float(payload.get("sinkhorn_reg") if payload.get("sinkhorn_reg") is not None else 1.0),
        )


def _log_optimal_transport(
    scores: torch.Tensor,
    dustbin_score: torch.Tensor,
    *,
    num_iters: int,
    reg: float,
) -> torch.Tensor:
    """Differentiable balanced token-to-cluster assignment with a dustbin."""

    if scores.ndim != 3:
        raise ValueError("scores must be [batch, tokens, clusters]")
    batch, token_count, cluster_count = scores.shape
    if token_count <= 0 or cluster_count <= 0:
        raise ValueError("scores must contain at least one token and cluster")
    work = scores.float().transpose(1, 2) / max(1e-6, float(reg))
    dust = dustbin_score.float().view(1, 1, 1).expand(batch, 1, token_count)
    work = torch.cat([work, dust], dim=1)
    rows = cluster_count + 1
    cols = token_count
    norm = -torch.tensor(float(rows + cols), device=work.device, dtype=work.dtype).log()
    log_a = norm.expand(batch, rows).clone()
    log_b = norm.expand(batch, cols).clone()
    log_a[:, -1] = log_a[:, -1] + torch.tensor(max(1.0, float(cols - cluster_count)), device=work.device, dtype=work.dtype).log()
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(max(1, int(num_iters))):
        u = log_a - torch.logsumexp(work + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(work + u.unsqueeze(2), dim=1)
    return work + u.unsqueeze(2) + v.unsqueeze(1) - norm


class LocalSALADHead(nn.Module):
    """Trainable SALAD-style head for DINO patch tokens."""

    def __init__(self, config: LocalSALADConfig):
        super().__init__()
        if int(config.num_channels) <= 0:
            raise ValueError("num_channels must be positive")
        self.config = config
        dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.token_features = nn.Sequential(
            nn.Linear(config.num_channels, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Linear(config.num_channels, config.hidden_dim),
            dropout,
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.cluster_dim),
        )
        self.score = nn.Sequential(
            nn.Linear(config.num_channels, config.hidden_dim),
            dropout,
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.num_clusters),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    @property
    def output_dim(self) -> int:
        return int(self.config.token_dim + self.config.num_clusters * self.config.cluster_dim)

    def forward(self, patch_tokens: torch.Tensor, global_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must be [batch, tokens, channels]")
        if patch_tokens.shape[-1] != self.config.num_channels:
            raise ValueError(
                f"patch token width {patch_tokens.shape[-1]} does not match head width {self.config.num_channels}"
            )
        patches = torch.nan_to_num(patch_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if global_token is None:
            global_work = patches.mean(dim=1)
        else:
            global_work = torch.nan_to_num(global_token.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if global_work.ndim == 3:
                global_work = global_work[:, 0, :]
            if global_work.ndim != 2 or global_work.shape[-1] != self.config.num_channels:
                global_work = patches.mean(dim=1)
        local_features = self.cluster_features(patches).transpose(1, 2)
        scores = self.score(patches)
        log_assign = _log_optimal_transport(
            scores,
            self.dust_bin,
            num_iters=self.config.sinkhorn_iters,
            reg=self.config.sinkhorn_reg,
        )
        assign = log_assign[:, :-1, :].exp()
        pooled = torch.einsum("bdn,bkn->bdk", local_features, assign)
        pooled = F.normalize(pooled, p=2, dim=1).reshape(patches.shape[0], -1)
        global_desc = F.normalize(self.token_features(global_work), p=2, dim=-1)
        return F.normalize(torch.cat([global_desc, pooled], dim=-1), p=2, dim=-1)


def symmetric_infonce_loss(left: torch.Tensor, right: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left and right descriptors must have matching [batch, dim] shapes")
    if left.shape[0] <= 1:
        raise ValueError("InfoNCE needs at least two paired samples")
    temp = max(1e-4, float(temperature))
    logits = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T / temp
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def load_local_salad_head_file(path: Path, *, device_name: str = "cpu") -> Tuple[LocalSALADHead, Dict[str, Any]]:
    """Load a Tator-local SALAD head saved by the local training path."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"local SALAD head not found: {resolved}")
    try:
        payload = torch.load(resolved, map_location=device_name, weights_only=True)
    except TypeError:
        payload = torch.load(resolved, map_location=device_name)
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
    head = LocalSALADHead(config)
    head.load_state_dict(payload.get("state_dict") or {})
    head.to(device_name)
    head.eval()
    return head, {"path": str(resolved), **metadata, "config": config.to_dict()}
