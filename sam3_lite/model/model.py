from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetBackbone
from .head import MLP
from .position import PositionEmbeddingSine
from .transformer import TransformerDecoder


class Sam3LiteModel(nn.Module):
    def __init__(
        self,
        num_queries: int = 200,
        num_classes: int = 1,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_decoder_layers: int = 6,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(out_dim=hidden_dim, pretrained=pretrained_backbone)
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer_decoder = TransformerDecoder(num_layers=num_decoder_layers, d_model=hidden_dim, nhead=nheads)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.presence_head = MLP(hidden_dim, hidden_dim, num_classes + 1, num_layers=3)  # extra background class

    def forward(self, images: torch.Tensor):
        # images: [B,3,H,W]
        feats = self.backbone(images)  # [B,C,Hf,Wf]
        pos = self.position_embedding(feats)
        b, c, h, w = feats.shape
        src = feats.flatten(2).permute(0, 2, 1)  # [B,HW,C]
        pos_flat = pos.flatten(2).permute(0, 2, 1)  # [B,HW,C]
        memory = src + pos_flat
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        hs = self.transformer_decoder(query_pos, memory)
        pred_boxes = self.bbox_embed(hs).sigmoid()
        pred_logits = self.presence_head(hs)
        return {
            "pred_boxes": pred_boxes,  # [B,num_queries,4] in cxcywh normalized
            "pred_logits": pred_logits,  # class logits incl. background
        }


def build_model(num_classes: int, num_queries: int = 200, pretrained_backbone: bool = False) -> Sam3LiteModel:
    return Sam3LiteModel(num_queries=num_queries, num_classes=num_classes, pretrained_backbone=pretrained_backbone)
