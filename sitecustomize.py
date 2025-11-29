"""
Lightweight runtime monkeypatches for SAM3 training.

We prefer not to edit the vendor package; when the environment variable
SAM3_MONKEYPATCH=1 is present we tweak the SAM3 trainer at import time.
"""

import logging
import os
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


if os.environ.get("SAM3_MONKEYPATCH") == "1":
    _patch_sam3_trainer()
