from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BALANCE_STRATEGIES = {"none", "inv_sqrt", "clipped_inv", "effective_num", "focal"}


@dataclass
class PathsConfig:
    train_img_folder: str
    train_ann_file: str
    val_img_folder: str
    val_ann_file: str
    bpe_path: Optional[str] = None
    signature: Optional[str] = None
    init_checkpoint: Optional[str] = None

    @property
    def train_root(self) -> Path:
        return Path(self.train_img_folder)

    @property
    def val_root(self) -> Path:
        return Path(self.val_img_folder)


@dataclass
class DatasetConfig:
    class_balance: bool = False
    balance_strategy: str = "none"
    balance_power: float = 0.5
    balance_clip: float = 10.0
    balance_beta: float = 0.99
    balance_gamma: float = 0.5
    classes: List[str] = field(default_factory=list)
    train_limit: Optional[int] = None

    def normalized(self) -> "DatasetConfig":
        strategy = (self.balance_strategy or "none").lower()
        if strategy not in BALANCE_STRATEGIES:
            strategy = "none"
        return replace(self, balance_strategy=strategy)


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_train_workers: int = 4
    num_val_workers: int = 2
    gradient_accumulation_steps: int = 1
    target_epoch_size: int = 1000
    val_epoch_freq: int = 10
    lr_scale: float = 1.0
    scheduler_warmup: int = 20
    scheduler_timescale: int = 20
    resolution: int = 1008
    enable_inst_interactivity: bool = False
    num_gpus: int = 1
    log_freq: int = 10
    seed: int = 123


@dataclass
class LauncherConfig:
    num_nodes: int = 1
    gpus_per_node: int = 1


@dataclass
class MetadataConfig:
    dataset_id: Optional[str] = None
    created_at: Optional[float] = None


@dataclass
class RunConfig:
    run_name: str
    experiment_log_dir: str
    paths: PathsConfig
    dataset: DatasetConfig
    trainer: TrainerConfig
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "experiment_log_dir": self.experiment_log_dir,
            "paths": self.paths.__dict__,
            "dataset": self.dataset.__dict__,
            "trainer": self.trainer.__dict__,
            "launcher": self.launcher.__dict__,
            "metadata": self.metadata.__dict__,
        }


def _deep_update(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
    raw: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    if overrides:
        raw = _deep_update(raw, overrides)

    run_name = raw.get("run_name") or path.stem
    exp_dir = raw.get("experiment_log_dir") or str(path.parent)

    paths_raw = raw.get("paths") or {}
    dataset_raw = raw.get("dataset") or {}
    trainer_raw = raw.get("trainer") or {}
    launcher_raw = raw.get("launcher") or {}
    metadata_raw = raw.get("metadata") or {}

    paths = PathsConfig(
        train_img_folder=paths_raw.get("train_img_folder", ""),
        train_ann_file=paths_raw.get("train_ann_file", ""),
        val_img_folder=paths_raw.get("val_img_folder", ""),
        val_ann_file=paths_raw.get("val_ann_file", ""),
        bpe_path=paths_raw.get("bpe_path"),
        signature=paths_raw.get("signature"),
    )
    dataset = DatasetConfig(**dataset_raw).normalized()
    trainer = TrainerConfig(**trainer_raw)
    launcher = LauncherConfig(**launcher_raw)
    metadata = MetadataConfig(**metadata_raw)

    return RunConfig(
        run_name=run_name,
        experiment_log_dir=exp_dir,
        paths=paths,
        dataset=dataset,
        trainer=trainer,
        launcher=launcher,
        metadata=metadata,
    )
