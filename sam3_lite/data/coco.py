from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import Sample


def _coco_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


class CocoDetection(Dataset):
    def __init__(
        self,
        ann_file: Path | str,
        img_folder: Path | str,
        transforms: Optional[Any] = None,
        train_limit: Optional[int] = None,
    ):
        self.ann_file = Path(ann_file)
        self.img_folder = Path(img_folder)
        self.transforms = transforms

        with self.ann_file.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        images = raw.get("images", [])
        annotations = raw.get("annotations", [])
        categories = raw.get("categories", [])
        cat_id_to_name = {c["id"]: c["name"] for c in categories}
        sorted_ids = sorted(cat_id_to_name.keys())
        self.cat_id_to_idx = {cid: idx for idx, cid in enumerate(sorted_ids)}
        self.idx_to_cat = {idx: cid for cid, idx in self.cat_id_to_idx.items()}
        self.categories = {cid: cat_id_to_name[cid] for cid in sorted_ids}
        self.classes = [cat_id_to_name[cid] for cid in sorted_ids]

        if train_limit is not None and train_limit > 0:
            images = images[:train_limit]

        self.images = images
        self.annotations = annotations
        self.ann_index: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            if ann.get("iscrowd"):
                continue
            img_id = ann["image_id"]
            self.ann_index.setdefault(img_id, []).append(ann)

        self.image_labels: List[List[int]] = []
        for img in self.images:
            anns = self.ann_index.get(img["id"], [])
            labels = sorted({int(self.cat_id_to_idx[int(a["category_id"])]) for a in anns if int(a["category_id"]) in self.cat_id_to_idx})
            self.image_labels.append(labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Sample:
        img_info = self.images[idx]
        img_path = self.img_folder / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        anns = self.ann_index.get(img_info["id"], [])
        boxes = torch.tensor([_coco_to_xyxy(a["bbox"]) for a in anns], dtype=torch.float32)
        labels = torch.tensor(
            [int(self.cat_id_to_idx[int(a["category_id"])]) for a in anns if int(a["category_id"]) in self.cat_id_to_idx],
            dtype=torch.int64,
        )
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        sample: Sample = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(img_info["id"], dtype=torch.int64),
            "orig_size": torch.tensor([img_info.get("height", image.height), img_info.get("width", image.width)], dtype=torch.int64),
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


def collate_fn(batch: List[Sample]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets: List[Dict[str, Any]] = []
    for b in batch:
        targets.append(
            {
                "boxes": b["boxes"],
                "labels": b["labels"],
                "image_id": b["image_id"],
                "orig_size": b.get("orig_size"),
                "size": b.get("size"),
            }
        )
    return {"images": images, "targets": targets}
