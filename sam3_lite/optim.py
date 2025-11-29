from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _get_parameter_groups(model: nn.Module, weight_decay: float = 0.01) -> List[dict]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    param_groups = _get_parameter_groups(model, weight_decay=weight_decay)
    return AdamW(param_groups, lr=lr, betas=(0.9, 0.999))


def build_inverse_sqrt_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int = 1000, timescale: int = 2000) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        step = max(1, step)
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        return (timescale ** 0.5) / (step ** 0.5)

    return LambdaLR(optimizer, lr_lambda)
