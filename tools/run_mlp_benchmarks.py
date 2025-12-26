#!/usr/bin/env python3
"""Run a repeatable MLP benchmark sweep and update metrics tables."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.clip_training import TrainingError, train_clip_from_yolo  # noqa: E402


def _post_form(api_root: str, path: str, data: Dict[str, str]) -> Dict[str, Any]:
    url = f"{api_root.rstrip('/')}{path}"
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = resp.read()
    return json.loads(payload.decode("utf-8"))


def _get_json(api_root: str, path: str) -> Dict[str, Any]:
    url = f"{api_root.rstrip('/')}{path}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = resp.read()
    return json.loads(payload.decode("utf-8"))


def _poll_job(api_root: str, job_id: str, poll_interval: float = 10.0) -> Dict[str, Any]:
    while True:
        status = _get_json(api_root, f"/clip/train/{job_id}")
        state = status.get("status")
        if state in {"succeeded", "failed", "cancelled"}:
            return status
        time.sleep(poll_interval)


def _format_sizes(sizes: Sequence[int]) -> str:
    return "_".join(str(int(s)) for s in sizes)


def _parse_sizes(value: Sequence[int]) -> str:
    return ",".join(str(int(s)) for s in value)


def _metric_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _compute_metrics(
    per_class_metrics: Sequence[Dict[str, Optional[float]]],
    *,
    fg_only: bool,
) -> Dict[str, Optional[float]]:
    filtered: List[Dict[str, Optional[float]]] = []
    for row in per_class_metrics:
        label = str(row.get("label", ""))
        if fg_only and label.startswith("__bg_"):
            continue
        filtered.append(row)
    if not filtered:
        return {
            "macro_f1": None,
            "weighted_f1": None,
            "macro_precision": None,
            "macro_recall": None,
        }
    f1_vals: List[float] = []
    prec_vals: List[float] = []
    rec_vals: List[float] = []
    weighted_sum = 0.0
    weighted_total = 0.0
    for row in filtered:
        f1 = row.get("f1")
        prec = row.get("precision")
        rec = row.get("recall")
        support = row.get("support") or 0
        if f1 is not None:
            f1_vals.append(float(f1))
        if prec is not None:
            prec_vals.append(float(prec))
        if rec is not None:
            rec_vals.append(float(rec))
        if f1 is not None and support:
            weighted_sum += float(f1) * float(support)
            weighted_total += float(support)
    weighted_f1 = (weighted_sum / weighted_total) if weighted_total else None
    return {
        "macro_f1": _metric_mean(f1_vals),
        "weighted_f1": weighted_f1,
        "macro_precision": _metric_mean(prec_vals),
        "macro_recall": _metric_mean(rec_vals),
    }


def _load_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, ensure_ascii=True)


def _write_csv(path: Path, payload: List[Dict[str, Any]]) -> None:
    if not payload:
        return
    fields = [
        "label",
        "encoder_type",
        "encoder_model",
        "embedding_dim",
        "accuracy",
        "macro_f1_all",
        "macro_f1_fg",
        "weighted_f1_all",
        "weighted_f1_fg",
        "iters",
        "converged",
        "model_path",
        "labelmap_path",
        "meta_path",
        "job_id",
        "macro_precision_fg",
        "macro_recall_fg",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload:
            writer.writerow({key: row.get(key) for key in fields})


def _write_summary(path: Path, payload: List[Dict[str, Any]]) -> None:
    lines = [
        "Comparison summary (accuracy + F1, FG-only excludes __bg_* classes)",
        "",
        "label, encoder_model, acc, macro_f1_fg, weighted_f1_fg",
    ]
    sorted_rows = sorted(
        payload,
        key=lambda entry: (entry.get("macro_f1_fg") or 0.0, entry.get("accuracy") or 0.0),
        reverse=True,
    )
    for row in sorted_rows:
        acc = row.get("accuracy")
        macro_fg = row.get("macro_f1_fg")
        weighted_fg = row.get("weighted_f1_fg")
        acc_str = f"{acc:.4f}" if isinstance(acc, (float, int)) else "n/a"
        macro_str = f"{macro_fg:.4f}" if isinstance(macro_fg, (float, int)) else "n/a"
        weighted_str = f"{weighted_fg:.4f}" if isinstance(weighted_fg, (float, int)) else "n/a"
        lines.append(
            f"{row.get('label')}, {row.get('encoder_model')}, {acc_str}, {macro_str}, {weighted_str}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_experiments(dataset_name: str, bg_classes: int) -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = []
    dinov3 = [
        (
            "vits16",
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            [[128], [192], [256], [384], [256, 128], [384, 192]],
        ),
        (
            "vitb16",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            [[256], [384], [512], [768], [512, 256], [768, 384], [768, 384, 192]],
        ),
        (
            "vitl16",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            [[512], [768], [1024], [768, 384], [1024, 512], [1024, 512, 256]],
        ),
        (
            "vitl16_sat493m",
            "facebook/dinov3-vitl16-pretrain-sat493m",
            [[768, 384], [1024, 512]],
        ),
        (
            "vit7b16_sat493m",
            "facebook/dinov3-vit7b16-pretrain-sat493m",
            [[1536, 768], [2048, 1024]],
        ),
    ]
    clip = [
        ("vitb32", "ViT-B/32", [[256], [512, 256]]),
        ("vitb16", "ViT-B/16", [[256], [512, 256]]),
        ("vitl14", "ViT-L/14", [[512], [768, 384]]),
    ]
    for smoothing in (0.0, 0.1):
        ls_tag = "0p1" if smoothing > 0 else "0"
        for short, model, sizes_list in dinov3:
            for sizes in sizes_list:
                label = (
                    f"{dataset_name}_dinov3_{short}_mlp{_format_sizes(sizes)}"
                    f"_mix0p1_balnorm_ls{ls_tag}_bg{bg_classes}"
                )
                experiments.append({
                    "label": label,
                    "encoder_type": "dinov3",
                    "encoder_model": model,
                    "clip_model": model,
                    "mlp_sizes": sizes,
                    "label_smoothing": smoothing,
                    "bg_classes": bg_classes,
                })
        for short, model, sizes_list in clip:
            for sizes in sizes_list:
                label = (
                    f"{dataset_name}_clip_{short}_mlp{_format_sizes(sizes)}"
                    f"_mix0p1_balnorm_ls{ls_tag}_bg{bg_classes}"
                )
                experiments.append({
                    "label": label,
                    "encoder_type": "clip",
                    "encoder_model": model,
                    "clip_model": model,
                    "mlp_sizes": sizes,
                    "label_smoothing": smoothing,
                    "bg_classes": bg_classes,
                })
    return experiments


def _batch_size_for_model(encoder_type: str, model_name: str) -> int:
    name = (model_name or "").lower()
    if encoder_type == "dinov3":
        if "vit7b" in name:
            return 4
        if "vith" in name or "vitl" in name:
            return 16
        if "vitb" in name:
            return 32
        return 64
    if "vit-l/14" in name:
        return 32
    if "vit-b/16" in name:
        return 64
    if "vit-b/32" in name:
        return 64
    return 64


def _run_api_experiment(
    api_root: str,
    exp: Dict[str, Any],
    *,
    images_path: str,
    labels_path: str,
    labelmap_path: str,
    device: Optional[str],
    reuse_embeddings: bool,
) -> Dict[str, Any]:
    label = exp["label"]
    model_filename = f"{label}.pkl"
    labelmap_filename = f"{label}_labels.pkl"
    encoder_type = exp["encoder_type"]
    model_name = exp["encoder_model"]
    batch_size = _batch_size_for_model(encoder_type, model_name)
    data = {
        "images_path_native": images_path,
        "labels_path_native": labels_path,
        "labelmap_path_native": labelmap_path,
        "encoder_type": encoder_type,
        "encoder_model": model_name,
        "clip_model_name": model_name,
        "model_filename": model_filename,
        "labelmap_filename": labelmap_filename,
        "classifier_type": "mlp",
        "mlp_hidden_sizes": _parse_sizes(exp["mlp_sizes"]),
        "mlp_dropout": "0.1",
        "mlp_epochs": "50",
        "mlp_lr": "0.001",
        "mlp_weight_decay": "0.0001",
        "mlp_label_smoothing": f"{exp['label_smoothing']}",
        "mlp_loss_type": "ce",
        "mlp_sampler": "balanced",
        "mlp_mixup_alpha": "0.1",
        "mlp_normalize_embeddings": "true",
        "class_weight": "balanced",
        "bg_class_count": str(exp["bg_classes"]),
        "batch_size": str(batch_size),
        "random_seed": "42",
        "test_size": "0.2",
        "min_per_class": "2",
    }
    if reuse_embeddings:
        data["reuse_embeddings"] = "true"
    if device:
        data["device_override"] = device
    response = _post_form(api_root, "/clip/train", data)
    job_id = response.get("job_id")
    if not job_id:
        raise RuntimeError(f"Training job missing job_id for {label}")
    status = _poll_job(api_root, job_id)
    if status.get("status") != "succeeded":
        raise RuntimeError(f"Training job {job_id} failed: {status.get('error') or status.get('message')}")
    artifacts = status.get("artifacts") or {}
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CLIP/DINOv3 MLP benchmark sweep.")
    parser.add_argument("--dataset-name", default="qwen_dataset")
    parser.add_argument("--images-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--labelmap-path", required=True)
    parser.add_argument("--metrics-base", default="clip_dinov3_metrics_20241224")
    parser.add_argument("--bg-classes", type=int, default=5)
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--api-root", default=None, help="If set, run via backend API (e.g., http://127.0.0.1:8000).")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    metrics_base = Path(args.metrics_base)
    json_path = metrics_base.with_suffix(".json")
    csv_path = metrics_base.with_suffix(".csv")
    txt_path = metrics_base.with_suffix(".txt")

    payload = _load_metrics(json_path)
    seen_labels = {row.get("label") for row in payload}
    experiments = build_experiments(args.dataset_name, args.bg_classes)

    os.makedirs("uploads/classifiers", exist_ok=True)
    os.makedirs("uploads/labelmaps", exist_ok=True)

    for exp in experiments:
        label = exp["label"]
        if args.skip_existing and label in seen_labels:
            print(f"[skip] {label}")
            continue
        model_path = Path("uploads/classifiers") / f"{label}.pkl"
        labelmap_path = Path("uploads/labelmaps") / f"{label}_labels.pkl"
        print(f"[run] {label}")
        try:
            if args.api_root:
                artifacts = _run_api_experiment(
                    args.api_root,
                    exp,
                    images_path=args.images_path,
                    labels_path=args.labels_path,
                    labelmap_path=args.labelmap_path,
                    device=args.device,
                    reuse_embeddings=args.reuse_embeddings,
                )
            else:
                artifacts = train_clip_from_yolo(
                    images_path=args.images_path,
                    labels_path=args.labels_path,
                    model_output=str(model_path),
                    labelmap_output=str(labelmap_path),
                    input_labelmap=args.labelmap_path,
                    clip_model=exp["clip_model"],
                    encoder_type=exp["encoder_type"],
                    encoder_model=exp["encoder_model"] if exp["encoder_type"] == "dinov3" else None,
                    classifier_type="mlp",
                    mlp_hidden_sizes=_parse_sizes(exp["mlp_sizes"]),
                    mlp_label_smoothing=exp["label_smoothing"],
                    mlp_loss_type="ce",
                    mlp_sampler="balanced",
                    mlp_mixup_alpha=0.1,
                    mlp_normalize_embeddings=True,
                    class_weight="balanced",
                    bg_class_count=exp["bg_classes"],
                    reuse_embeddings=args.reuse_embeddings,
                    device=args.device,
                )
        except TrainingError as exc:
            print(f"[fail] {label}: {exc}")
            continue
        except Exception as exc:
            print(f"[fail] {label}: {exc}")
            continue

        per_class_metrics = artifacts.get("per_class_metrics") if isinstance(artifacts, dict) else artifacts.per_class_metrics
        if not per_class_metrics:
            print(f"[fail] {label}: missing per-class metrics in artifacts")
            continue
        all_metrics = _compute_metrics(per_class_metrics, fg_only=False)
        fg_metrics = _compute_metrics(per_class_metrics, fg_only=True)
        job_id = uuid.uuid4().hex
        row = {
            "label": label,
            "encoder_type": artifacts.get("encoder_type") if isinstance(artifacts, dict) else artifacts.encoder_type,
            "encoder_model": artifacts.get("encoder_model") if isinstance(artifacts, dict) else artifacts.encoder_model,
            "embedding_dim": artifacts.get("embedding_dim") if isinstance(artifacts, dict) else artifacts.embedding_dim,
            "accuracy": artifacts.get("accuracy") if isinstance(artifacts, dict) else artifacts.accuracy,
            "macro_f1_all": all_metrics["macro_f1"],
            "macro_f1_fg": fg_metrics["macro_f1"],
            "weighted_f1_all": all_metrics["weighted_f1"],
            "weighted_f1_fg": fg_metrics["weighted_f1"],
            "iters": artifacts.get("iterations_run") if isinstance(artifacts, dict) else artifacts.iterations_run,
            "converged": artifacts.get("converged") if isinstance(artifacts, dict) else artifacts.converged,
            "model_path": artifacts.get("model_path") if isinstance(artifacts, dict) else artifacts.model_path,
            "labelmap_path": artifacts.get("labelmap_path") if isinstance(artifacts, dict) else artifacts.labelmap_path,
            "meta_path": artifacts.get("meta_path") if isinstance(artifacts, dict) else artifacts.meta_path,
            "job_id": job_id,
            "macro_precision_fg": fg_metrics["macro_precision"],
            "macro_recall_fg": fg_metrics["macro_recall"],
        }
        payload.append(row)
        seen_labels.add(label)

        _write_json(json_path, payload)
        _write_csv(csv_path, payload)
        _write_summary(txt_path, payload)

    print(f"Saved metrics to {json_path}, {csv_path}, {txt_path}")


if __name__ == "__main__":
    main()
