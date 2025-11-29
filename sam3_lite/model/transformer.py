from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = k = tgt
        tgt2 = self.self_attn(q, k, tgt, need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(tgt)
