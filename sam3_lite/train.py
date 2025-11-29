from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import torch

from sam3_lite.config import load_config
from sam3_lite.train_loop import train
from sam3_lite.utils import setup_logging


def _install_signal_handlers():
    stop = {"flag": False}

    def _handler(signum, frame):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    return stop


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3-lite trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--log-dir", default=None, help="Directory for logs")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs requested")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    exp_dir = Path(cfg.experiment_log_dir).resolve()
    log_dir = Path(args.log_dir) if args.log_dir else exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir)

    stop = _install_signal_handlers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_gpus > 1 and not torch.cuda.is_available():
        logger.warning("Requested %s GPUs but CUDA is not available. Falling back to CPU.", args.num_gpus)
    logger.info("Loaded config from %s", cfg_path)
    logger.info("Using device=%s", device)

    if stop["flag"]:
        logger.info("Cancelled before start")
        return 1

    train(cfg, log_dir, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
