from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import torch
import torchvision.transforms.functional as F


Sample = Dict[str, torch.Tensor]


class Compose:
    def __init__(self, transforms: Sequence[Callable[[Sample], Sample]]):
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


@dataclass
class ResizeAndPad:
    size: int
    square: bool = True

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        _, h, w = image.shape
        target = self.size
        scale = target / float(max(h, w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        resized = F.resize(image, [new_h, new_w], antialias=True)
        boxes = sample.get("boxes")
        if boxes is not None:
            boxes = boxes * torch.tensor([scale, scale, scale, scale], device=boxes.device, dtype=boxes.dtype)
        if self.square:
            pad_h = target - new_h
            pad_w = target - new_w
            top = pad_h // 2
            left = pad_w // 2
            resized = F.pad(resized, [left, top, pad_w - left, pad_h - top], fill=0)
            if boxes is not None:
                boxes = boxes + torch.tensor([left, top, left, top], device=boxes.device, dtype=boxes.dtype)
        sample["image"] = resized
        if boxes is not None:
            sample["boxes"] = boxes
        sample["size"] = torch.tensor([resized.shape[1], resized.shape[2]], dtype=torch.int64)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        image = sample["image"]
        _, h, w = image.shape
        flipped = torch.flip(image, dims=[2])
        boxes = sample.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            x_min = boxes[:, 0].clone()
            x_max = boxes[:, 2].clone()
            boxes[:, 0] = w - x_max
            boxes[:, 2] = w - x_min
            sample["boxes"] = boxes
        sample["image"] = flipped
        return sample


class RandomBBoxJitter:
    def __init__(self, box_noise_std: float = 0.1, box_noise_max: float = 20.0):
        self.box_noise_std = box_noise_std
        self.box_noise_max = box_noise_max

    def __call__(self, sample: Sample) -> Sample:
        boxes = sample.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return sample
        noise = torch.randn_like(boxes) * self.box_noise_std
        noise = noise.clamp(min=-self.box_noise_max, max=self.box_noise_max)
        jittered = boxes + noise
        orig_size = sample.get("orig_size")
        if orig_size is not None:
            h, w = int(orig_size[0]), int(orig_size[1])
            jittered[:, 0::2].clamp_(min=0, max=w)
            jittered[:, 1::2].clamp_(min=0, max=h)
        sample["boxes"] = jittered
        return sample


class ToTensor:
    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        if not torch.is_tensor(image):
            image = F.to_tensor(image)
        sample["image"] = image
        return sample


class Normalize:
    def __init__(self, mean: List[float] | torch.Tensor, std: List[float] | torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        sample["image"] = F.normalize(image, mean=self.mean, std=self.std)
        return sample


class ClampBoxes:
    def __call__(self, sample: Sample) -> Sample:
        boxes = sample.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return sample
        h, w = sample.get("size") or sample.get("orig_size")
        boxes[:, 0::2].clamp_(min=0, max=float(w))
        boxes[:, 1::2].clamp_(min=0, max=float(h))
        sample["boxes"] = boxes
        return sample
