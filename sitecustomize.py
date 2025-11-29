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
                # Avoid DDP unused parameter bookkeeping when segmentation is frozen.
                try:
                    if getattr(self, "distributed_conf", None) is not None:
                        self.distributed_conf.find_unused_parameters = False
                except Exception:
                    pass
                logging.info(
                    "Monkeypatch: segmentation head params frozen and marked to ignore in DDP "
                    "because load_segmentation=False (params: %s)",
                    seg_param_names,
                )
        return out

    sam3_trainer.Trainer._setup_components = _setup_components_patched


if os.environ.get("SAM3_MONKEYPATCH") == "1":
    _patch_sam3_trainer()
