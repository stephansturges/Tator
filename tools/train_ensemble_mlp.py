#!/usr/bin/env python
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def _split_by_image(meta_rows: List[Dict[str, str]], seed: int, val_ratio: float) -> Dict[str, List[int]]:
    images = sorted({row["image"] for row in meta_rows if row.get("image")})
    if not images:
        return {"train": [], "val": []}
    rng = random.Random(seed)
    image_labels: Dict[str, set] = {img: set() for img in images}
    for row in meta_rows:
        img = row.get("image")
        if not img:
            continue
        label = str(row.get("label") or "").strip()
        if label:
            image_labels.setdefault(img, set()).add(label)
    label_images: Dict[str, List[str]] = {}
    for img, labels in image_labels.items():
        for label in labels:
            label_images.setdefault(label, []).append(img)

    val_target = max(1, int(len(images) * val_ratio))
    val_images: set = set()

    for label, img_list in sorted(label_images.items(), key=lambda item: len(item[1])):
        if not img_list:
            continue
        desired = max(1, int(len(img_list) * val_ratio))
        current = len([img for img in img_list if img in val_images])
        if current >= desired:
            continue
        candidates = [img for img in img_list if img not in val_images]
        rng.shuffle(candidates)
        for img in candidates:
            val_images.add(img)
            current += 1
            if current >= desired:
                break

    remaining = [img for img in images if img not in val_images]
    rng.shuffle(remaining)
    for img in remaining:
        if len(val_images) >= val_target:
            break
        val_images.add(img)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for idx, row in enumerate(meta_rows):
        if row.get("image") in val_images:
            val_idx.append(idx)
        else:
            train_idx.append(idx)
    return {"train": train_idx, "val": val_idx}


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], dropout: float) -> None:
        super().__init__()
        layers: List[torch.nn.Module] = []
        last = input_dim
        for width in hidden:
            layers.append(torch.nn.Linear(last, width))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            last = width
        layers.append(torch.nn.Linear(last, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class BalancedBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(
        self,
        targets: np.ndarray,
        *,
        batch_size: int,
        pos_fraction: float,
        seed: int,
        pos_threshold: float,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.pos_fraction = float(max(0.0, min(1.0, pos_fraction)))
        self.rng = np.random.default_rng(seed)
        self.pos_idx = np.where(targets >= pos_threshold)[0]
        self.neg_idx = np.where(targets < pos_threshold)[0]
        self.num_batches = max(1, int(math.ceil(len(targets) / float(self.batch_size))))
        if len(self.pos_idx) == 0:
            self.pos_count = 0
            self.neg_count = self.batch_size
        elif len(self.neg_idx) == 0:
            self.pos_count = self.batch_size
            self.neg_count = 0
        else:
            desired = int(round(self.batch_size * self.pos_fraction))
            self.pos_count = max(1, min(desired, self.batch_size - 1))
            self.neg_count = self.batch_size - self.pos_count

    def __iter__(self):
        for _ in range(self.num_batches):
            batch: List[int] = []
            if self.pos_count > 0:
                batch.extend(self.rng.choice(self.pos_idx, size=self.pos_count, replace=True).tolist())
            if self.neg_count > 0:
                batch.extend(self.rng.choice(self.neg_idx, size=self.neg_count, replace=True).tolist())
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def _compute_class_weights(
    labels: List[str],
    y: np.ndarray,
    *,
    mode: str,
    neg_mode: str,
    pos_threshold: float,
) -> np.ndarray:
    if mode == "none":
        return np.ones_like(y, dtype=np.float32)
    pos_counts: Dict[str, int] = {}
    neg_counts: Dict[str, int] = {}
    for label, target in zip(labels, y):
        if target >= pos_threshold:
            pos_counts[label] = pos_counts.get(label, 0) + 1
        else:
            neg_counts[label] = neg_counts.get(label, 0) + 1
    pos_vals = [v for v in pos_counts.values() if v > 0]
    neg_vals = [v for v in neg_counts.values() if v > 0]
    median_pos = float(np.median(pos_vals)) if pos_vals else 1.0
    median_neg = float(np.median(neg_vals)) if neg_vals else 1.0
    total_pos = float(sum(pos_vals))
    total_neg = float(sum(neg_vals))
    global_pos_weight = total_neg / max(total_pos, 1.0)

    weights = np.ones_like(y, dtype=np.float32)
    for idx, (label, target) in enumerate(zip(labels, y)):
        if mode == "global":
            weights[idx] = float(global_pos_weight) if target > 0 else 1.0
            continue
        if target > 0:
            denom = max(float(pos_counts.get(label, 0)), 1.0)
            weights[idx] = float(median_pos) / denom
        else:
            if neg_mode == "none":
                weights[idx] = 1.0
            else:
                denom = max(float(neg_counts.get(label, 0)), 1.0)
                ratio = float(median_neg) / denom
                if neg_mode == "sqrt":
                    weights[idx] = float(np.sqrt(max(ratio, 0.0)))
                else:
                    weights[idx] = ratio
    return weights


def _compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_type: str,
    sample_weights: torch.Tensor,
    focal_gamma: float,
    focal_alpha: float,
    asym_gamma_pos: float,
    asym_gamma_neg: float,
    pos_threshold: float,
    soft_targets: bool,
    tversky_alpha: float,
    tversky_beta: float,
    focal_tversky_gamma: float,
) -> torch.Tensor:
    if loss_type == "bce":
        base = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        return (base * sample_weights).mean()

    probs = torch.sigmoid(logits)
    if loss_type in ("tversky", "focal_tversky"):
        tp = (sample_weights * probs * targets).sum()
        fp = (sample_weights * probs * (1.0 - targets)).sum()
        fn = (sample_weights * (1.0 - probs) * targets).sum()
        smooth = 1e-6
        denom = tp + tversky_alpha * fn + tversky_beta * fp + smooth
        tversky = (tp + smooth) / denom
        loss = 1.0 - tversky
        if loss_type == "focal_tversky":
            loss = torch.pow(loss, focal_tversky_gamma)
        return loss
    base = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    alpha = float(focal_alpha)
    if soft_targets:
        alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        gamma = targets * asym_gamma_pos + (1.0 - targets) * asym_gamma_neg
    else:
        pos_mask = targets >= pos_threshold
        alpha_t = torch.where(pos_mask, alpha, 1.0 - alpha)
        pt = torch.where(pos_mask, probs, 1.0 - probs)
        gamma = torch.where(pos_mask, asym_gamma_pos, asym_gamma_neg)
    if loss_type == "focal":
        focal_term = torch.pow(1.0 - pt, focal_gamma)
        loss = alpha_t * focal_term * base
        return (loss * sample_weights).mean()

    focal_term = torch.pow(1.0 - pt, gamma)
    loss = alpha_t * focal_term * base
    return (loss * sample_weights).mean()

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP accept/reject model.")
    parser.add_argument("--input", required=True, help="Input labeled .npz file.")
    parser.add_argument("--output", required=True, help="Output model prefix (no extension).")
    parser.add_argument("--hidden", default="256,128", help="Comma-separated hidden sizes.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--loss", default="bce", choices=["bce", "focal", "asym_focal", "tversky", "focal_tversky"], help="Loss function.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument("--focal-alpha", type=float, default=-1.0, help="Focal loss alpha (negative = auto).")
    parser.add_argument("--asym-gamma-pos", type=float, default=1.0, help="Asymmetric focal gamma for positives.")
    parser.add_argument("--asym-gamma-neg", type=float, default=4.0, help="Asymmetric focal gamma for negatives.")
    parser.add_argument("--tversky-alpha", type=float, default=0.7, help="Tversky alpha (FN weight).")
    parser.add_argument("--tversky-beta", type=float, default=0.3, help="Tversky beta (FP weight).")
    parser.add_argument("--focal-tversky-gamma", type=float, default=1.0, help="Focal Tversky gamma.")
    parser.add_argument("--class-balance", default="global", choices=["none", "global", "per_class"], help="Class balance weighting.")
    parser.add_argument("--neg-weight-mode", default="sqrt", choices=["none", "linear", "sqrt"], help="Negative weight mode for per-class balancing.")
    parser.add_argument("--sampler", default="none", choices=["none", "weighted"], help="Sampling strategy.")
    parser.add_argument("--batch-balance", default="none", choices=["none", "fixed"], help="Balanced batch sampling.")
    parser.add_argument("--pos-fraction", type=float, default=0.5, help="Positive fraction for balanced batches.")
    parser.add_argument("--pos-threshold", type=float, default=0.5, help="Positive threshold for weighting/sampling.")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Early stop patience (0 disables).")
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine", "step"], help="LR scheduler.")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Min LR for cosine scheduler.")
    parser.add_argument("--step-size", type=int, default=10, help="Step size for step scheduler.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for step scheduler.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda).")
    parser.add_argument("--target-mode", default="hard", choices=["hard", "iou"], help="Target mode (hard labels or IoU).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.input, allow_pickle=True)
    X = data["X"].astype(np.float32)
    if args.target_mode == "iou":
        if "y_iou" not in data:
            raise SystemExit("Missing y_iou in labeled dataset.")
        y = data["y_iou"].astype(np.float32)
    else:
        y = data["y"].astype(np.float32)
    meta_raw = list(data["meta"])
    meta = [json.loads(str(row)) for row in meta_raw]
    feature_names = list(data["feature_names"])
    labelmap = list(data.get("labelmap", []))
    classifier_classes = list(data.get("classifier_classes", []))
    sam3_iou = float(data.get("sam3_iou", 0.5))
    label_iou = float(data.get("label_iou", 0.9))

    split = _split_by_image(meta, seed=args.seed, val_ratio=args.val_ratio)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise SystemExit("Not enough samples for train/val split.")
    val_images = sorted({meta[idx].get("image") for idx in val_idx if meta[idx].get("image")})
    train_images = sorted({meta[idx].get("image") for idx in train_idx if meta[idx].get("image")})

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32)
    labels_train = [meta[idx].get("label", "") for idx in train_idx]
    labels_val = [meta[idx].get("label", "") for idx in val_idx]

    hidden = [int(h.strip()) for h in args.hidden.split(",") if h.strip()]
    model = MLP(X.shape[1], hidden, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(args.step_size)), gamma=float(args.gamma))

    pos = float((y_train >= float(args.pos_threshold)).sum().item())
    neg = float((y_train < float(args.pos_threshold)).sum().item())
    if args.focal_alpha < 0:
        alpha = min(0.95, max(0.05, neg / max(pos + neg, 1.0)))
    else:
        alpha = float(args.focal_alpha)

    train_weights_np = _compute_class_weights(
        labels_train,
        y_train.numpy(),
        mode=args.class_balance,
        neg_mode=args.neg_weight_mode,
        pos_threshold=float(args.pos_threshold),
    )
    val_weights_np = _compute_class_weights(
        labels_val,
        y_val.numpy(),
        mode=args.class_balance,
        neg_mode=args.neg_weight_mode,
        pos_threshold=float(args.pos_threshold),
    )
    train_weights = torch.tensor(train_weights_np, dtype=torch.float32)
    val_weights = torch.tensor(val_weights_np, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, train_weights)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val, val_weights)
    sampler = None
    batch_sampler = None
    if args.batch_balance != "none":
        batch_sampler = BalancedBatchSampler(
            y_train.numpy(),
            batch_size=int(args.batch_size),
            pos_fraction=float(args.pos_fraction),
            seed=int(args.seed),
            pos_threshold=float(args.pos_threshold),
        )
    elif args.sampler == "weighted":
        sampler = torch.utils.data.WeightedRandomSampler(
            train_weights_np,
            num_samples=len(train_weights_np),
            replacement=True,
        )
    if batch_sampler is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            sampler=sampler,
            shuffle=(sampler is None),
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
    )

    best_val = None
    best_state = None
    patience = int(args.early_stop_patience)
    patience_left = patience
    soft_targets = args.target_mode != "hard"
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = max(1, int(args.grad_accum))
        total_loss = 0.0
        steps = 0
        for step, (xb, yb, wb) in enumerate(train_loader, start=1):
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            wb = wb.to(args.device)
            logits = model(xb)
            loss = _compute_loss(
                logits,
                yb,
                loss_type=args.loss,
                sample_weights=wb,
                focal_gamma=float(args.focal_gamma),
                focal_alpha=float(alpha),
                asym_gamma_pos=float(args.asym_gamma_pos),
                asym_gamma_neg=float(args.asym_gamma_neg),
                pos_threshold=float(args.pos_threshold),
                soft_targets=soft_targets,
                tversky_alpha=float(args.tversky_alpha),
                tversky_beta=float(args.tversky_beta),
                focal_tversky_gamma=float(args.focal_tversky_gamma),
            )
            (loss / accum).backward()
            total_loss += float(loss.item())
            steps += 1
            if step % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if steps % accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss_total = 0.0
            val_steps = 0
            for xb, yb, wb in val_loader:
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                wb = wb.to(args.device)
                val_logits = model(xb)
                val_loss = _compute_loss(
                    val_logits,
                    yb,
                    loss_type=args.loss,
                    sample_weights=wb,
                    focal_gamma=float(args.focal_gamma),
                    focal_alpha=float(alpha),
                    asym_gamma_pos=float(args.asym_gamma_pos),
                    asym_gamma_neg=float(args.asym_gamma_neg),
                    pos_threshold=float(args.pos_threshold),
                    soft_targets=soft_targets,
                    tversky_alpha=float(args.tversky_alpha),
                    tversky_beta=float(args.tversky_beta),
                    focal_tversky_gamma=float(args.focal_tversky_gamma),
                ).item()
                val_loss_total += float(val_loss)
                val_steps += 1
            val_loss = val_loss_total / max(val_steps, 1)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if patience:
                patience_left = patience
        elif patience:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early_stop epoch={epoch}", flush=True)
                break
        avg_loss = total_loss / max(steps, 1)
        print(f"epoch={epoch} loss={avg_loss:.4f} val_loss={val_loss:.4f}", flush=True)

    model_path = Path(args.output).with_suffix(".pt")
    meta_path = Path(args.output).with_suffix(".meta.json")
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X.shape[1]),
            "hidden": hidden,
            "dropout": float(args.dropout),
        },
        model_path,
    )

    meta_out = {
        "feature_names": feature_names,
        "labelmap": labelmap,
        "classifier_classes": classifier_classes,
        "sam3_iou": sam3_iou,
        "label_iou": label_iou,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "split_train_images": train_images,
        "split_val_images": val_images,
        "val_loss": float(best_val or 0.0),
        "split_seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "model_path": str(model_path),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": str(args.scheduler),
        "min_lr": float(args.min_lr),
        "step_size": int(args.step_size),
        "gamma": float(args.gamma),
        "loss": str(args.loss),
        "focal_gamma": float(args.focal_gamma),
        "focal_alpha": float(alpha),
        "asym_gamma_pos": float(args.asym_gamma_pos),
        "asym_gamma_neg": float(args.asym_gamma_neg),
        "tversky_alpha": float(args.tversky_alpha),
        "tversky_beta": float(args.tversky_beta),
        "focal_tversky_gamma": float(args.focal_tversky_gamma),
        "class_balance": str(args.class_balance),
        "neg_weight_mode": str(args.neg_weight_mode),
        "sampler": str(args.sampler),
        "batch_balance": str(args.batch_balance),
        "pos_fraction": float(args.pos_fraction),
        "pos_threshold": float(args.pos_threshold),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "early_stop_patience": int(args.early_stop_patience),
        "target_mode": str(args.target_mode),
    }
    meta_path.write_text(json.dumps(meta_out, indent=2))


if __name__ == "__main__":
    main()
