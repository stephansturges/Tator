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
            seg_param_names = [
                name
                for name, _ in getattr(self.model, "named_parameters", lambda: [])()
                if "segmentation_head" in name
            ]
            if seg_param_names:
                # Track the ignored params so DDP setup can be adjusted later.
                self._tator_ignore_params = seg_param_names
                self._tator_force_disable_find_unused = True
                for name, param in self.model.named_parameters():
                    if name in seg_param_names:
                        param.requires_grad = False
                # DDP matches names before wrapping, but be safe and include module-prefixed variants.
                ddp_ignore = set(
                    getattr(self.model, "_ddp_params_and_buffers_to_ignore", []) or []
                )
                ddp_ignore.update(seg_param_names)
                ddp_ignore.update([f"module.{n}" for n in seg_param_names])
                self.model._ddp_params_and_buffers_to_ignore = list(ddp_ignore)
                logging.info(
                    "Monkeypatch: segmentation head params frozen and marked to ignore in DDP "
                    "because load_segmentation=False (params: %s)",
                    seg_param_names,
                )
        return out

    def _setup_ddp_distributed_training_patched(
        self: Any, distributed_conf: Any, accelerator: Any
    ):
        # If we froze/ignored segmentation params, disable unused-parameter
        # bookkeeping so DDP does not expect gradients for them.
        if getattr(self, "_tator_force_disable_find_unused", False):
            if distributed_conf.find_unused_parameters:
                logging.info(
                    "Monkeypatch: forcing find_unused_parameters=False because %d params are ignored.",
                    len(getattr(self, "_tator_ignore_params", [])),
                )
            distributed_conf.find_unused_parameters = False
        return original_setup_ddp(self, distributed_conf, accelerator)

    sam3_trainer.Trainer._setup_components = _setup_components_patched
    sam3_trainer.Trainer._setup_ddp_distributed_training = (
        _setup_ddp_distributed_training_patched
    )


if os.environ.get("SAM3_MONKEYPATCH") == "1":
    _patch_sam3_trainer()
