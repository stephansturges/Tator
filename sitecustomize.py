"""
Lightweight runtime monkeypatches for SAM3 training.

We prefer not to edit the vendor package; by default we tweak the SAM3 trainer
at import time. Set SAM3_MONKEYPATCH=0 to disable.
"""

import logging
import os
from collections import deque
from typing import Any


def _patch_sam3_trainer() -> None:
    try:
        from sam3.train import trainer as sam3_trainer
    except Exception:
        return

    if getattr(sam3_trainer, "_tator_monkeypatched", False):
        return

    sam3_trainer._tator_monkeypatched = True
    original_setup = sam3_trainer.Trainer._setup_components
    original_setup_ddp = sam3_trainer.Trainer._setup_ddp_distributed_training
    original_print_model_summary = sam3_trainer.print_model_summary

    def _setup_components_patched(self: Any):
        # Run the original setup first.
        out = original_setup(self)

        # Determine if segmentation masks are being loaded.
        load_segmentation = False
        try:
            load_segmentation = bool(
                self.data_conf.train.dataset.get("load_segmentation", False)
            )
        except Exception:
            pass

        if not load_segmentation and hasattr(self, "model"):
            ignore_param_names = []
            for name, _ in getattr(self.model, "named_parameters", lambda: [])():
                if "segmentation_head" in name:
                    ignore_param_names.append(name)
                elif "backbone.vision_backbone.convs" in name:
                    # Convs block feeds segmentation features; not needed for bbox-only.
                    ignore_param_names.append(name)
            if ignore_param_names:
                self._tator_ignore_params = ignore_param_names
                self._tator_force_disable_find_unused = True
                for name, param in self.model.named_parameters():
                    if name in ignore_param_names:
                        param.requires_grad = False
                ddp_ignore = set(
                    getattr(self.model, "_ddp_params_and_buffers_to_ignore", []) or []
                )
                ddp_ignore.update(ignore_param_names)
                ddp_ignore.update([f"module.{n}" for n in ignore_param_names])
                self.model._ddp_params_and_buffers_to_ignore = list(ddp_ignore)
                logging.info(
                    "Monkeypatch: froze and ignored params not needed for bbox-only training: %s",
                    ignore_param_names,
                )
        return out

    def _setup_ddp_distributed_training_patched(
        self: Any, distributed_conf: Any, accelerator: Any
    ):
        return original_setup_ddp(self, distributed_conf, accelerator)

    def _print_model_summary_patched(model: Any, log_dir: str = "") -> None:
        """
        Avoid dumping the entire model repr, but still log parameter counts.
        """
        if getattr(model, "_tator_summary_printed", False):
            return
        model._tator_summary_printed = True
        if sam3_trainer.get_rank() != 0:
            return
        param_kwargs = {}
        trainable_parameters = sum(
            p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
        )
        total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
        non_trainable_parameters = total_parameters - trainable_parameters
        logging.info("==" * 10)
        logging.info(f"Summary for model {type(model)}")
        logging.info(
            "\tParam counts (trainable / total / frozen): "
            f"{trainable_parameters:,} / {total_parameters:,} / {non_trainable_parameters:,}"
        )
        logging.info("==" * 10)
        if log_dir:
            try:
                with sam3_trainer.g_pathmgr.open(
                    os.path.join(log_dir, "model.txt"), "w"
                ) as f:
                    print(model, file=f)
            except Exception:
                pass

    sam3_trainer.Trainer._setup_components = _setup_components_patched
    sam3_trainer.Trainer._setup_ddp_distributed_training = (
        _setup_ddp_distributed_training_patched
    )
    sam3_trainer.print_model_summary = _print_model_summary_patched


def _monkeypatch_enabled() -> bool:
    flag = os.environ.get("SAM3_MONKEYPATCH", "1").lower()
    return flag not in ("0", "false", "off", "no")


def _patch_logging_smoothing() -> None:
    try:
        from sam3.train.utils import train_utils as tu
    except Exception:
        return
    if getattr(tu, "_tator_logging_patch", False):
        return
    tu._tator_logging_patch = True

    class RollingAverageMeter(tu.AverageMeter):
        """
        Drop-in replacement that computes a rolling, sample-weighted average for losses.
        Defaults to window=50 for loss meters; retains cumulative avg for others.
        """

        def __init__(self, name, device, fmt=":f", window_size=None):
            self.window_size = window_size
            if self.window_size is None and isinstance(name, str) and "Losses/" in name:
                self.window_size = 50
            self._window = deque(maxlen=self.window_size) if self.window_size else None
            super().__init__(name, device, fmt)

        def reset(self):
            super().reset()
            self._window = deque(maxlen=self.window_size) if self.window_size else None

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            if self._window is not None:
                self._window.append((val, n))
                win_sum = sum(v * w for v, w in self._window)
                win_count = sum(w for _, w in self._window)
                self.avg = win_sum / max(win_count, 1)
            else:
                self.avg = self.sum / self.count

        def __str__(self):
            fmt_spec = self.fmt.lstrip(":")
            if self.window_size:
                return f"{self.name}: last={self.val:{fmt_spec}} avg{self.window_size}={self.avg:{fmt_spec}}"
            return f"{self.name}: last={self.val:{fmt_spec}} avg={self.avg:{fmt_spec}}"

    tu.AverageMeter = RollingAverageMeter
    tu.RollingAverageMeter = RollingAverageMeter

    # Trim noisy progress logging: drop data/mem meters and extra real_meters section.
    orig_display = tu.ProgressMeter.display

    def _display_filtered(self, batch, enable_print=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        for meter in self.meters:
            name = getattr(meter, "name", "")
            if name in {"Data Time", "Mem (GB)"}:
                continue
            entries.append(str(meter))
        msg = " | ".join(entries)
        logging.info(msg)
        if enable_print:
            print(msg)

    tu.ProgressMeter.display = _display_filtered


if _monkeypatch_enabled():
    _patch_sam3_trainer()
    _patch_logging_smoothing()
else:
    logging.info("SAM3 monkeypatch disabled via SAM3_MONKEYPATCH=0")
