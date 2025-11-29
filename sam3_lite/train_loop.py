from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sam3_lite.config import RunConfig
from sam3_lite.data import (
    ClampBoxes,
    CocoDetection,
    Compose,
    Normalize,
    RandomBBoxJitter,
    RandomHorizontalFlip,
    ResizeAndPad,
    ToTensor,
    build_sampler,
    collate_fn,
)
from sam3_lite.losses import DetectionLoss
from sam3_lite.model.model import build_model
from sam3_lite.optim import build_inverse_sqrt_scheduler, build_optimizer
from sam3_lite.utils import setup_logging


def _build_dataloaders(cfg: RunConfig):
    resize = ResizeAndPad(cfg.trainer.resolution, square=True)
    common_transforms = Compose([ToTensor(), resize, ClampBoxes(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    jitter = RandomBBoxJitter(0.1, 20.0)
    flip = RandomHorizontalFlip(0.5)

    train_ds = CocoDetection(cfg.paths.train_ann_file, cfg.paths.train_img_folder, transforms=Compose([flip, jitter, common_transforms]), train_limit=cfg.dataset.train_limit)
    val_ds = CocoDetection(cfg.paths.val_ann_file, cfg.paths.val_img_folder, transforms=common_transforms)

    epoch_size = cfg.trainer.target_epoch_size or len(train_ds)
    sampler, stats = build_sampler(train_ds.image_labels, cfg.dataset, epoch_size=epoch_size)
    if stats:
        # emit a quick summary for logs
        print(f"[sam3lite-balance] classes={len(train_ds.classes)} min_w={stats.get('min'):.4f} avg_w={stats.get('avg'):.4f} max_w={stats.get('max'):.4f}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.trainer.train_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.trainer.num_train_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.trainer.val_batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_val_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def _save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.ckpt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, ckpt_path)
    return ckpt_path


def train(cfg: RunConfig, log_dir: Path, device: torch.device) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir)
    logger.info("Starting SAM3-lite run=%s device=%s", cfg.run_name, device)

    torch.manual_seed(cfg.trainer.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.trainer.seed)

    train_loader, val_loader = _build_dataloaders(cfg)

    num_classes = max(1, len(train_loader.dataset.classes)) if hasattr(train_loader, "dataset") else 1  # type: ignore[attr-defined]
    model = build_model(num_classes=num_classes, num_queries=200, pretrained_backbone=True)

    # Optional fine-tune from checkpoint
    init_ckpt = cfg.paths.init_checkpoint
    if not init_ckpt:
        # try env override first
        init_ckpt = os.environ.get("SAM3_LITE_INIT_CKPT")
    if not init_ckpt:
        # auto-discover a vendor checkpoint if present
        candidate_dir = Path("sam3")
        if candidate_dir.exists():
            for p in sorted(candidate_dir.rglob("*.pth")):
                if "sam3" in p.name.lower():
                    init_ckpt = str(p)
                    break
    if init_ckpt:
        ckpt_path = Path(init_ckpt)
        if ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                state_dict = state.get("model", state)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded init checkpoint %s (missing=%d unexpected=%d)", ckpt_path, len(missing), len(unexpected))
            except Exception as exc:
                logger.warning("Failed to load init checkpoint %s: %s", ckpt_path, exc)
        else:
            logger.warning("Init checkpoint not found: %s", ckpt_path)
    model.to(device)

    # learning rate scaled by lr_scale vs default
    base_lr = 1e-4 * float(cfg.trainer.lr_scale)
    optimizer = build_optimizer(model, lr=base_lr, weight_decay=0.01)
    scheduler = build_inverse_sqrt_scheduler(optimizer, warmup_steps=cfg.trainer.scheduler_warmup, timescale=cfg.trainer.scheduler_timescale)
    criterion = DetectionLoss()

    step = 0
    best_ckpt: Optional[Path] = None
    best_val: Optional[float] = None
    ckpt_dir = Path(cfg.experiment_log_dir) / "checkpoints"
    start_time = time.time()

    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=log_dir / "tensorboard")
    except Exception:
        tb_writer = None

    for epoch in range(cfg.trainer.max_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in batch["targets"]]
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses["loss_total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            step += 1

            progress = min(0.99, step / float(cfg.trainer.target_epoch_size)) if cfg.trainer.target_epoch_size else 0.0
            if step % cfg.trainer.log_freq == 0 or batch_idx == 0:
                loss_val = float(loss.item())
                msg = f"epoch={epoch} step={step} loss={loss_val:.4f}"
                logger.info(msg)
                print(f"[sam3lite-progress {progress:.4f}] {msg}", flush=True)
                metric_payload = {"step": step, "loss": loss_val, "progress": progress}
                import json  # local import to avoid top-level cost

                print(f"[sam3lite-metric]{json.dumps(metric_payload)}", flush=True)

            if cfg.trainer.target_epoch_size and step >= cfg.trainer.target_epoch_size:
                break

        # simple val loop every val_epoch_freq
        if (epoch + 1) % max(1, cfg.trainer.val_epoch_freq) == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for batch in val_loader:
                    images = batch["images"].to(device)
                    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in batch["targets"]]
                    outputs = model(images)
                    losses = criterion(outputs, targets)
                    val_losses.append(float(losses["loss_total"].item()))
                if val_losses:
                    val_mean = sum(val_losses) / len(val_losses)
                    logger.info("val_loss=%.4f", val_mean)
                    import json

                    print(f"[sam3lite-metric]{json.dumps({'step': step, 'val_loss': val_mean})}", flush=True)
                    if tb_writer:
                        tb_writer.add_scalar("val/loss", val_mean, step)
                    if best_val is None or val_mean < best_val:
                        best_val = val_mean
                        best_ckpt = _save_checkpoint(model, optimizer, step, ckpt_dir)

        last_ckpt = _save_checkpoint(model, optimizer, step, ckpt_dir)
        if tb_writer:
            tb_writer.add_scalar("train/loss", float(loss.item()), step)
        if cfg.trainer.target_epoch_size and step >= cfg.trainer.target_epoch_size:
            break

    elapsed = time.time() - start_time
    result = {
        "run_name": cfg.run_name,
        "experiment_log_dir": str(cfg.experiment_log_dir),
        "checkpoint": str(best_ckpt or last_ckpt),
        "step": step,
        "elapsed_sec": elapsed,
    }
    import json

    print(f"[sam3lite-result]{json.dumps(result)}", flush=True)
    if tb_writer:
        tb_writer.close()
    return result
