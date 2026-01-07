import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_mlp_benchmarks import _run_api_experiment, _compute_metrics

exp = {
    "label": "qwen_dataset_dinov3_vitl16_mlp1024_512_mix0p1_balnorm_ls0p1_full_logit_both_default_bg5_confirm",
    "encoder_type": "dinov3",
    "encoder_model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "clip_model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "mlp_sizes": [1024, 512],
    "label_smoothing": 0.1,
    "bg_classes": 5,
    "embedding_standardize": True,
    "class_weight": "effective",
    "effective_beta": 0.9999,
    "hard_example_mining": True,
    "mlp_activation": "gelu",
    "mlp_layer_norm": True,
    "calibration_mode": "temperature",
    "logit_adjustment_mode": "both",
}

artifacts = _run_api_experiment(
    "http://127.0.0.1:8000",
    exp,
    images_path="/home/steph/Tator/uploads/qwen_runs/datasets/qwen_dataset",
    labels_path="/home/steph/Tator/uploads/clip_dataset_uploads/qwen_dataset_yolo/labels",
    labelmap_path="/home/steph/Tator/uploads/clip_dataset_uploads/qwen_dataset_yolo/labelmap.txt",
    device="cuda:0",
    reuse_embeddings=True,
)

per_class = artifacts.get("per_class_metrics", {}) if isinstance(artifacts, dict) else artifacts.per_class_metrics
all_metrics = _compute_metrics(per_class, fg_only=False)
fg_metrics = _compute_metrics(per_class, fg_only=True)
print("confirm run:", all_metrics)
print("confirm fg:", fg_metrics)

import json
from pathlib import Path
out = {
    "label": exp["label"],
    "all": all_metrics,
    "fg": fg_metrics,
}
Path("default_recipe_confirm.json").write_text(json.dumps(out, indent=2, sort_keys=True))
print("wrote default_recipe_confirm.json")
