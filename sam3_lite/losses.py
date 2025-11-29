from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment

from sam3_lite.model.postprocess import box_cxcywh_to_xyxy


def _cxcywh_to_xyxy_norm(boxes: torch.Tensor) -> torch.Tensor:
    return box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)


def simple_greedy_match(
    pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor
) -> Tuple[List[int], List[int]]:
    if tgt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return [], []
    pred_xyxy = _cxcywh_to_xyxy_norm(pred_boxes)
    tgt_xyxy = _cxcywh_to_xyxy_norm(tgt_boxes)
    ious = generalized_box_iou(pred_xyxy, tgt_xyxy)  # [P,T]
    matches_pred: List[int] = []
    matches_tgt: List[int] = []
    used_pred = set()
    used_tgt = set()
    flat = torch.argsort(ious.flatten(), descending=True)
    for idx in flat.tolist():
        p = idx // ious.shape[1]
        t = idx % ious.shape[1]
        if p in used_pred or t in used_tgt:
            continue
        used_pred.add(p)
        used_tgt.add(t)
        matches_pred.append(int(p))
        matches_tgt.append(int(t))
    return matches_pred, matches_tgt


class DetectionLoss(nn.Module):
    def __init__(self, bbox_weight: float = 5.0, giou_weight: float = 2.0, cls_weight: float = 1.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.cls_weight = cls_weight

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pred_boxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]  # [B,Q,C+1] (bg last)

        device = pred_boxes.device
        total_bbox = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        total_matched = 0
        total_queries = 0

        for b, tgt in enumerate(targets):
            tgt_boxes = tgt["boxes"].to(device)
            size = tgt.get("size") or tgt.get("orig_size")
            if size is not None:
                # normalize to [0,1]
                w = size[1].float().clamp(min=1.0)
                h = size[0].float().clamp(min=1.0)
                scale = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
                tgt_boxes = tgt_boxes / scale
            if tgt_boxes.numel() == 0:
                # all negatives
                total_presence = total_presence + F.binary_cross_entropy_with_logits(pred_logits[b], torch.zeros_like(pred_logits[b]))
                total_queries += pred_logits.shape[1]
                continue
            preds_b = pred_boxes[b]
            logits_b = pred_logits[b]
            tgt_labels = tgt["labels"].to(device)
            num_classes = logits_b.shape[-1] - 1

            # Hungarian matching
            cost_class = -F.log_softmax(logits_b[:, :num_classes], dim=-1)[:, tgt_labels]  # [Q,T]
            cost_bbox = torch.cdist(preds_b, tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(_cxcywh_to_xyxy_norm(preds_b), _cxcywh_to_xyxy_norm(tgt_boxes))
            total_cost = self.bbox_weight * cost_bbox + self.giou_weight * cost_giou + self.cls_weight * cost_class
            row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu())
            matched_pred_idx = list(row_ind)
            matched_tgt_idx = list(col_ind)

            if matched_pred_idx:
                matched_preds = preds_b[matched_pred_idx]
                matched_tgts = tgt_boxes[matched_tgt_idx]
                matched_preds_xyxy = _cxcywh_to_xyxy_norm(matched_preds)
                matched_tgts_xyxy = _cxcywh_to_xyxy_norm(matched_tgts)
                l1 = F.l1_loss(matched_preds, matched_tgts, reduction="mean")
                giou = 1.0 - torch.diag(generalized_box_iou(matched_preds_xyxy, matched_tgts_xyxy)).mean()
                total_bbox = total_bbox + l1
                total_giou = total_giou + giou
                total_matched += len(matched_pred_idx)

            # class loss with background
            target_classes = torch.full((logits_b.shape[0],), fill_value=num_classes, device=device, dtype=torch.long)
            for p_idx, t_idx in zip(matched_pred_idx, matched_tgt_idx):
                target_classes[p_idx] = tgt_labels[t_idx]
            total_cls = total_cls + F.cross_entropy(logits_b, target_classes, reduction="mean")
            total_queries += logits_b.shape[0]

        losses: Dict[str, torch.Tensor] = {}
        if total_matched > 0:
            losses["loss_bbox"] = total_bbox * self.bbox_weight
            losses["loss_giou"] = total_giou * self.giou_weight
        else:
            losses["loss_bbox"] = total_bbox
            losses["loss_giou"] = total_giou
        if total_queries > 0:
            losses["loss_cls"] = total_cls * self.cls_weight
        else:
            losses["loss_cls"] = total_cls
        losses["loss_total"] = losses["loss_bbox"] + losses["loss_giou"] + losses["loss_cls"]
        return losses
