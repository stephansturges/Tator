from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Path, filename: str = "train.log", level: int = logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sam3_lite")
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    file_handler: Optional[logging.Handler] = None
    log_path = log_dir / filename
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path):
            file_handler = handler
            break
    if file_handler is None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
