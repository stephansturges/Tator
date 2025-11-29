from __future__ import annotations

from typing import Any, Dict, List

import torch


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h,
    ]
    return torch.stack(b, dim=-1)


def rescale_boxes(boxes: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    h, w = size.unbind(-1)
    scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
    return boxes * scale


def postprocess(outputs: Dict[str, torch.Tensor], sizes: torch.Tensor, topk: int = 100) -> List[Dict[str, Any]]:
    pred_boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
    pred_scores = outputs["pred_logits"].squeeze(-1).sigmoid()
    results: List[Dict[str, Any]] = []
    for boxes, scores, size in zip(pred_boxes, pred_scores, sizes):
        scores, idxs = scores.topk(min(topk, scores.numel()))
        boxes = boxes[idxs]
        boxes = rescale_boxes(boxes, size)
        results.append({"boxes": boxes, "scores": scores})
    return results
