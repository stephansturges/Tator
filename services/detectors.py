from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _set_yolo_infer_state_impl(
    model: Any,
    path: Optional[str],
    labelmap: List[str],
    task: Optional[str],
    *,
    state: Dict[str, Any],
) -> None:
    state["model"] = model
    state["path"] = path
    state["labelmap"] = labelmap
    state["task"] = task


def _set_rfdetr_infer_state_impl(
    model: Any,
    path: Optional[str],
    labelmap: List[str],
    task: Optional[str],
    variant: Optional[str],
    *,
    state: Dict[str, Any],
) -> None:
    state["model"] = model
    state["path"] = path
    state["labelmap"] = labelmap
    state["task"] = task
    state["variant"] = variant


def _ensure_yolo_inference_runtime_impl(
    *,
    load_active_fn: Callable[[], Dict[str, Any]],
    load_labelmap_fn: Callable[[Any], List[str]],
    patch_ultralytics_fn: Callable[[], None],
    yolo_lock: Any,
    get_state_fn: Callable[[], Tuple[Any, Optional[str], List[str], Optional[str]]],
    set_state_fn: Callable[[Any, Optional[str], List[str], Optional[str]], None],
    import_yolo_fn: Callable[[], Any],
    http_exception_cls: Any,
) -> Tuple[Any, List[str], Optional[str]]:
    active = load_active_fn()
    if not isinstance(active, dict):
        raise http_exception_cls(status_code=412, detail="yolo_active_missing")
    best_path = active.get("best_path")
    if not best_path:
        raise http_exception_cls(status_code=412, detail="yolo_active_missing")
    labelmap_path = active.get("labelmap_path")
    task = active.get("task")
    with yolo_lock:
        model, path, labelmap, cached_task = get_state_fn()
        if model is not None and path == best_path:
            return model, labelmap, cached_task
        try:
            YOLO = import_yolo_fn()
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(status_code=503, detail=f"yolo_unavailable:{exc}") from exc
        patch_ultralytics_fn()
        model = YOLO(best_path)
        labelmap = load_labelmap_fn(labelmap_path) if labelmap_path else []
        resolved_task = task or getattr(model, "task", None)
        set_state_fn(model, best_path, labelmap, resolved_task)
        return model, labelmap, resolved_task


def _ensure_rfdetr_inference_runtime_impl(
    *,
    load_active_fn: Callable[[], Dict[str, Any]],
    load_labelmap_fn: Callable[[Any], List[str]],
    variant_info_fn: Callable[[str, Optional[str]], Dict[str, Any]],
    rfdetr_lock: Any,
    get_state_fn: Callable[[], Tuple[Any, Optional[str], List[str], Optional[str], Optional[str]]],
    set_state_fn: Callable[[Any, Optional[str], List[str], Optional[str], Optional[str]], None],
    import_rfdetr_fn: Callable[[], Dict[str, Any]],
    http_exception_cls: Any,
    torch_available: Callable[[], bool],
    resolve_device_fn: Callable[[], str],
) -> Tuple[Any, List[str], Optional[str]]:
    active = load_active_fn()
    if not isinstance(active, dict):
        raise http_exception_cls(status_code=412, detail="rfdetr_active_missing")
    best_path = active.get("best_path")
    if not best_path:
        raise http_exception_cls(status_code=412, detail="rfdetr_active_missing")
    if not Path(best_path).exists():
        raise http_exception_cls(status_code=412, detail="rfdetr_active_missing_weights")
    labelmap_path = active.get("labelmap_path")
    task = active.get("task") or "detect"
    variant = active.get("variant")
    with rfdetr_lock:
        model, path, labelmap, cached_task, cached_variant = get_state_fn()
        if model is not None and path == best_path:
            return model, labelmap, cached_task
        try:
            import_map = import_rfdetr_fn()
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(status_code=503, detail=f"rfdetr_unavailable:{exc}") from exc
        variant_info = variant_info_fn(task, variant)
        variant_id = variant_info.get("id")
        model_cls = import_map.get(variant_id)
        if not model_cls:
            raise http_exception_cls(status_code=400, detail="rfdetr_variant_unknown")
        model_kwargs: Dict[str, Any] = {
            "pretrain_weights": best_path,
            "device": resolve_device_fn() if torch_available() else "cpu",
        }
        if variant_id == "rfdetr-seg-preview" or task == "segment":
            model_kwargs["segmentation_head"] = True
        model = model_cls(**model_kwargs)
        labelmap = load_labelmap_fn(labelmap_path) if labelmap_path else []
        if labelmap:
            try:
                model.model.class_names = labelmap
            except Exception:
                pass
        set_state_fn(model, best_path, labelmap, task, variant_id)
    return model, labelmap, task


def _load_yolo_active_impl(yolo_active_path: Path) -> Dict[str, Any]:
    if not yolo_active_path.exists():
        return {}
    try:
        return json.loads(yolo_active_path.read_text())
    except Exception:
        return {}


def _save_yolo_active_impl(payload: Dict[str, Any], yolo_active_path: Path) -> Dict[str, Any]:
    yolo_active_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload or {})
    data["updated_at"] = time.time()
    if "created_at" not in data:
        data["created_at"] = data["updated_at"]
    yolo_active_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def _load_rfdetr_active_impl(
    rfdetr_active_path: Path,
    rfdetr_job_root: Path,
    save_active_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    def _load_from(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    active = _load_from(rfdetr_active_path)
    best_path = str(active.get("best_path") or "")
    if best_path and Path(best_path).exists():
        return active

    fallback_path = rfdetr_job_root / "active.json"
    fallback = _load_from(fallback_path)
    fallback_best = str(fallback.get("best_path") or "")
    if fallback_best and Path(fallback_best).exists():
        try:
            save_active_fn(fallback)
        except Exception:
            pass
        return fallback
    return {}


def _save_rfdetr_active_impl(payload: Dict[str, Any], rfdetr_active_path: Path) -> Dict[str, Any]:
    rfdetr_active_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload or {})
    data["updated_at"] = time.time()
    if "created_at" not in data:
        data["created_at"] = data["updated_at"]
    rfdetr_active_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def _load_detector_default_impl(detector_default_path: Path) -> Dict[str, Any]:
    if not detector_default_path.exists():
        return {"mode": "rfdetr"}
    try:
        payload = json.loads(detector_default_path.read_text())
        if isinstance(payload, dict):
            mode = str(payload.get("mode") or "").strip().lower()
            if mode in {"yolo", "rfdetr"}:
                return payload
    except Exception:
        pass
    return {"mode": "rfdetr"}


def _save_detector_default_impl(
    payload: Dict[str, Any],
    detector_default_path: Path,
    http_exception_cls: Any,
) -> Dict[str, Any]:
    detector_default_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload or {})
    mode = str(data.get("mode") or "").strip().lower()
    if mode not in {"yolo", "rfdetr"}:
        raise http_exception_cls(status_code=400, detail="detector_mode_invalid")
    data["mode"] = mode
    data["updated_at"] = time.time()
    if "created_at" not in data:
        data["created_at"] = data["updated_at"]
    detector_default_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def _yolo_run_dir_impl(
    run_id: str,
    *,
    create: bool,
    job_root: Path,
    sanitize_fn: Callable[[str], str],
    http_exception_cls: Any,
) -> Path:
    safe_id = sanitize_fn(run_id)
    candidate = (job_root / safe_id).resolve()
    if not str(candidate).startswith(str(job_root.resolve())):
        raise http_exception_cls(status_code=400, detail="invalid_run_id")
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _yolo_load_run_meta_impl(run_dir: Path, *, meta_name: str) -> Dict[str, Any]:
    meta_path = run_dir / meta_name
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _yolo_write_run_meta_impl(
    run_dir: Path,
    meta: Dict[str, Any],
    *,
    meta_name: str,
    time_fn: Callable[[], float],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(meta or {})
    now = time_fn()
    payload.setdefault("created_at", now)
    payload["updated_at"] = now
    (run_dir / meta_name).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _yolo_prune_run_dir_impl(
    run_dir: Path,
    *,
    keep_files: Optional[set[str]],
    keep_files_default: Sequence[str],
    dir_size_fn: Callable[[Path], int],
    meta_name: str,
) -> Dict[str, Any]:
    kept: List[str] = []
    deleted: List[str] = []
    freed = 0
    if not run_dir.exists():
        return {"kept": kept, "deleted": deleted, "freed_bytes": freed}
    keep = set(keep_files or keep_files_default)
    for child in run_dir.iterdir():
        if child.is_file() and child.suffix == ".yaml":
            keep.add(child.name)
    weights_dir = run_dir / "weights"
    if weights_dir.exists():
        best_path = weights_dir / "best.pt"
        target_best = run_dir / "best.pt"
        if best_path.exists() and not target_best.exists():
            try:
                shutil.copy2(best_path, target_best)
            except Exception:
                pass
    for child in list(run_dir.iterdir()):
        if child.name in keep:
            kept.append(child.name)
            continue
        try:
            if child.is_dir():
                freed += dir_size_fn(child)
                shutil.rmtree(child)
            else:
                freed += child.stat().st_size
                child.unlink()
            deleted.append(child.name)
        except Exception:
            continue
    return {"kept": kept, "deleted": deleted, "freed_bytes": freed}


def _rfdetr_run_dir_impl(
    run_id: str,
    *,
    create: bool,
    job_root: Path,
    sanitize_fn: Callable[[str], str],
    http_exception_cls: Any,
) -> Path:
    safe_id = sanitize_fn(run_id)
    candidate = (job_root / safe_id).resolve()
    if not str(candidate).startswith(str(job_root.resolve())):
        raise http_exception_cls(status_code=400, detail="invalid_run_id")
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _rfdetr_load_run_meta_impl(run_dir: Path, *, meta_name: str) -> Dict[str, Any]:
    meta_path = run_dir / meta_name
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _rfdetr_write_run_meta_impl(
    run_dir: Path,
    meta: Dict[str, Any],
    *,
    meta_name: str,
    time_fn: Callable[[], float],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(meta or {})
    now = time_fn()
    payload.setdefault("created_at", now)
    payload["updated_at"] = now
    (run_dir / meta_name).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _rfdetr_prune_run_dir_impl(
    run_dir: Path,
    *,
    keep_files: Optional[set[str]],
    keep_files_default: Sequence[str],
    dir_size_fn: Callable[[Path], int],
) -> Dict[str, Any]:
    kept: List[str] = []
    deleted: List[str] = []
    freed = 0
    if not run_dir.exists():
        return {"kept": kept, "deleted": deleted, "freed_bytes": freed}
    keep = set(keep_files or keep_files_default)
    for child in list(run_dir.iterdir()):
        if child.name in keep:
            kept.append(child.name)
            continue
        try:
            if child.is_dir():
                freed += dir_size_fn(child)
                shutil.rmtree(child)
            else:
                freed += child.stat().st_size
                child.unlink()
            deleted.append(child.name)
        except Exception:
            continue
    return {"kept": kept, "deleted": deleted, "freed_bytes": freed}


def _yolo_device_arg_impl(devices: Optional[List[int]]) -> Optional[str]:
    if not devices:
        return None
    cleaned = [str(int(d)) for d in devices if isinstance(d, (int, str)) and str(d).strip().isdigit()]
    return ",".join(cleaned) if cleaned else None


def _yolo_p2_scale_impl(model_id: str) -> Optional[str]:
    match = re.match(r"^yolov8([nsmlx])-p2$", model_id)
    if match:
        return match.group(1)
    return None


def _rfdetr_variant_info_impl(
    task: str,
    variant: Optional[str],
    *,
    variants: Sequence[Dict[str, Any]],
    http_exception_cls: Any,
) -> Dict[str, Any]:
    task_norm = (task or "detect").lower().strip()
    variant_norm = (variant or "").strip().lower()
    if task_norm == "segment":
        variant_norm = "rfdetr-seg-preview"
    if not variant_norm:
        variant_norm = "rfdetr-medium" if task_norm == "detect" else "rfdetr-seg-preview"
    variant_map = {entry["id"]: entry for entry in variants}
    info = variant_map.get(variant_norm)
    if not info:
        raise http_exception_cls(status_code=400, detail="rfdetr_variant_unknown")
    if task_norm == "segment" and info.get("task") != "segment":
        variant_norm = "rfdetr-seg-preview"
        info = variant_map.get(variant_norm)
    return info or {}


def _rfdetr_best_checkpoint_impl(run_dir: Path) -> Optional[str]:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        path = run_dir / name
        if path.exists():
            return str(path)
    return None


def _rfdetr_parse_log_series_impl(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []
    series: List[Dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            series.append(json.loads(line))
        except Exception:
            continue
    return series


def _rfdetr_sanitize_metric_impl(metric: Dict[str, Any]) -> Dict[str, Any]:
    def _coerce(obj: Any) -> Any:
        if isinstance(obj, __import__("numpy").ndarray):
            return obj.tolist()
        if isinstance(obj, __import__("numpy").generic):
            try:
                return obj.item()
            except Exception:
                return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return obj

    try:
        return json.loads(json.dumps(metric, default=_coerce))
    except Exception:
        return {}


def _rfdetr_normalize_aug_policy_impl(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not raw or not isinstance(raw, dict):
        return None

    def _clamp(value: Any, default: float = 0.0, maximum: float = 1.0) -> float:
        try:
            num = float(value)
        except Exception:
            num = default
        return max(0.0, min(maximum, num))

    policy = {
        "hsv_h": _clamp(raw.get("hsv_h"), 0.0, 0.5),
        "hsv_s": _clamp(raw.get("hsv_s"), 0.0, 1.0),
        "hsv_v": _clamp(raw.get("hsv_v"), 0.0, 1.0),
        "blur_prob": _clamp(raw.get("blur_prob"), 0.0, 1.0),
        "gray_prob": _clamp(raw.get("gray_prob"), 0.0, 1.0),
    }
    kernel = raw.get("blur_kernel")
    try:
        kernel = int(kernel)
    except Exception:
        kernel = 0
    if kernel and kernel % 2 == 0:
        kernel += 1
    policy["blur_kernel"] = max(0, kernel)
    if not any(
        [
            policy["hsv_h"],
            policy["hsv_s"],
            policy["hsv_v"],
            policy["blur_prob"],
            policy["gray_prob"],
        ]
    ):
        return None
    return policy


def _rfdetr_install_augmentations_impl(
    policy: Optional[Dict[str, Any]],
) -> Optional[Tuple[Any, Any]]:
    if not policy:
        return None
    try:
        import random as _random
        import torchvision.transforms as tvt
        import rfdetr.datasets.coco as coco_mod
    except Exception:
        return None

    class _ImageOnlyTransform:
        def __init__(self, transform, p: float = 1.0) -> None:
            self.transform = transform
            self.p = float(p)

        def __call__(self, img, target):
            if self.p < 1.0 and _random.random() > self.p:
                return img, target
            return self.transform(img), target

    aug_transforms = []
    hsv_h = float(policy.get("hsv_h") or 0.0)
    hsv_s = float(policy.get("hsv_s") or 0.0)
    hsv_v = float(policy.get("hsv_v") or 0.0)
    if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
        color_jitter = tvt.ColorJitter(
            brightness=hsv_v,
            contrast=hsv_v,
            saturation=hsv_s,
            hue=min(0.5, hsv_h),
        )
        aug_transforms.append(_ImageOnlyTransform(color_jitter, p=1.0))
    gray_prob = float(policy.get("gray_prob") or 0.0)
    if gray_prob > 0:
        aug_transforms.append(_ImageOnlyTransform(tvt.RandomGrayscale(p=1.0), p=gray_prob))
    blur_prob = float(policy.get("blur_prob") or 0.0)
    blur_kernel = int(policy.get("blur_kernel") or 0)
    if blur_prob > 0 and blur_kernel >= 3:
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blur = tvt.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))
        aug_transforms.append(_ImageOnlyTransform(blur, p=blur_prob))
    if not aug_transforms:
        return None
    original_make = coco_mod.make_coco_transforms
    original_make_square = coco_mod.make_coco_transforms_square_div_64

    def _wrap_make(make_func):
        def _wrapped(*args, **kwargs):
            base = make_func(*args, **kwargs)
            try:
                if hasattr(base, "transforms") and isinstance(base.transforms, list):
                    insert_idx = max(0, len(base.transforms) - 1)
                    base.transforms[insert_idx:insert_idx] = list(aug_transforms)
            except Exception:
                pass
            return base
        return _wrapped

    coco_mod.make_coco_transforms = _wrap_make(original_make)
    coco_mod.make_coco_transforms_square_div_64 = _wrap_make(original_make_square)
    return (original_make, original_make_square)


def _rfdetr_restore_augmentations_impl(
    restore: Optional[Tuple[Any, Any]],
) -> None:
    if not restore:
        return
    try:
        import rfdetr.datasets.coco as coco_mod
    except Exception:
        return
    try:
        coco_mod.make_coco_transforms = restore[0]
        coco_mod.make_coco_transforms_square_div_64 = restore[1]
    except Exception:
        return


def _rfdetr_latest_checkpoint_epoch_impl(run_dir: Path) -> Optional[int]:
    try:
        best = None
        for path in run_dir.glob("checkpoint*.pth"):
            name = path.name
            if not name.startswith("checkpoint") or not name.endswith(".pth"):
                continue
            token = name[len("checkpoint") : -len(".pth")]
            if not token.isdigit():
                continue
            value = int(token)
            if best is None or value > best:
                best = value
        return best
    except Exception:
        return None


def _rfdetr_monitor_training_impl(
    job: Any,
    run_dir: Path,
    total_epochs: int,
    stop_event: Any,
    *,
    job_append_metric_fn: Callable[[Any, Dict[str, Any]], None],
    job_update_fn: Callable[[Any], None],
    sanitize_metric_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    latest_checkpoint_fn: Callable[[Path], Optional[int]],
    wait_seconds: float = 15.0,
) -> None:
    log_path = run_dir / "log.txt"
    last_pos = 0
    pending = ""
    last_epoch: Optional[int] = None
    while not stop_event.is_set():
        if job.cancel_event.is_set() or job.status not in {"running", "queued"}:
            break
        new_metrics: List[Dict[str, Any]] = []
        if log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    handle.seek(last_pos)
                    chunk = handle.read()
                    last_pos = handle.tell()
            except Exception:
                chunk = ""
            if chunk:
                chunk = pending + chunk
                pending = ""
                if not chunk.endswith("\n"):
                    last_newline = chunk.rfind("\n")
                    if last_newline == -1:
                        pending = chunk
                        chunk = ""
                    else:
                        pending = chunk[last_newline + 1 :]
                        chunk = chunk[: last_newline + 1]
                for line in chunk.splitlines():
                    if not line.strip():
                        continue
                    try:
                        metric = json.loads(line)
                    except Exception:
                        continue
                    metric = sanitize_metric_fn(metric)
                    if metric:
                        new_metrics.append(metric)
        if new_metrics:
            for metric in new_metrics:
                job_append_metric_fn(job, metric)
            latest = new_metrics[-1]
            epoch = latest.get("epoch")
            if isinstance(epoch, (int, float)):
                try:
                    epoch_idx = int(epoch)
                except Exception:
                    epoch_idx = None
                if epoch_idx is not None and epoch_idx != last_epoch:
                    last_epoch = epoch_idx
                    if total_epochs > 0:
                        progress = max(0.0, min(0.99, epoch_idx / total_epochs))
                        job_update_fn(job, progress=progress, message=f"Epoch {epoch_idx}/{total_epochs}")
        else:
            checkpoint_epoch = latest_checkpoint_fn(run_dir)
            if checkpoint_epoch is not None and checkpoint_epoch != last_epoch:
                last_epoch = checkpoint_epoch
                if total_epochs > 0:
                    progress = max(0.0, min(0.99, checkpoint_epoch / total_epochs))
                    job_update_fn(job, progress=progress, message=f"Epoch {checkpoint_epoch}/{total_epochs}")
    stop_event.wait(wait_seconds)


def _yolo_build_aug_args_impl(aug: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not aug:
        return {}
    payload = dict(aug)
    mapping = {
        "flip_lr": "fliplr",
        "flip_ud": "flipud",
        "hsv_h": "hsv_h",
        "hsv_s": "hsv_s",
        "hsv_v": "hsv_v",
        "mosaic": "mosaic",
        "mixup": "mixup",
        "copy_paste": "copy_paste",
        "scale": "scale",
        "translate": "translate",
        "degrees": "degrees",
        "shear": "shear",
        "perspective": "perspective",
        "erasing": "erasing",
    }
    aug_args: Dict[str, Any] = {}
    for key, dest in mapping.items():
        if key in payload:
            aug_args[dest] = payload[key]
    return {k: v for k, v in aug_args.items() if v is not None}


def _yolo_parse_results_csv_impl(results_path: Path) -> List[Dict[str, Any]]:
    if not results_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, raw in enumerate(reader):
                if not raw:
                    continue
                parsed: Dict[str, Any] = {}
                for key, value in raw.items():
                    if key is None or value is None:
                        continue
                    name = str(key).strip()
                    if not name:
                        continue
                    text = str(value).strip()
                    if text == "":
                        continue
                    try:
                        num = float(text)
                    except ValueError:
                        continue
                    if name == "epoch":
                        parsed["epoch"] = int(num) if float(num).is_integer() else num
                    else:
                        parsed[name] = num
                if "epoch" not in parsed:
                    parsed["epoch"] = idx + 1
                if parsed:
                    rows.append(parsed)
    except Exception:  # noqa: BLE001
        return []
    return rows


def _yolo_monitor_training_impl(
    job: Any,
    run_dir: Path,
    total_epochs: int,
    stop_event: Any,
    *,
    parse_results_fn: Callable[[Path], List[Dict[str, Any]]],
    job_append_metric_fn: Callable[[Any, Dict[str, Any]], None],
    job_update_fn: Callable[[Any], None],
    wait_seconds: float = 12.0,
) -> None:
    results_path = run_dir / "train" / "results.csv"
    last_len = 0
    while not stop_event.is_set():
        if job.cancel_event.is_set() or job.status not in {"running", "queued"}:
            break
        series = parse_results_fn(results_path)
        if series and len(series) > last_len:
            new_entries = series[last_len:]
            for metric in new_entries:
                job_append_metric_fn(job, metric)
            last_len = len(series)
            latest = series[-1]
            epoch = latest.get("epoch")
            if isinstance(epoch, (int, float)):
                try:
                    epoch_idx = int(epoch)
                except Exception:
                    epoch_idx = None
                if epoch_idx is not None and total_epochs > 0:
                    progress = max(0.0, min(0.99, epoch_idx / total_epochs))
                    job_update_fn(job, progress=progress, message=f"Epoch {epoch_idx}/{total_epochs}")
    stop_event.wait(wait_seconds)


def _strip_checkpoint_optimizer_impl(
    ckpt_path: Path,
    *,
    torch_module: Any,
) -> Tuple[bool, int, int]:
    """Remove optimizer/scheduler state from a torch checkpoint to shrink size."""
    before = ckpt_path.stat().st_size if ckpt_path.exists() else 0
    if not ckpt_path.exists() or before == 0:
        return False, before, before
    try:
        payload = torch_module.load(ckpt_path, map_location="cpu")
        removed = False
        for key in ["optimizer", "optimizers", "lr_schedulers", "schedulers", "trainer"]:
            if key in payload:
                payload.pop(key, None)
                removed = True
        if not removed:
            return False, before, before
        tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch_module.save(payload, tmp_path)
        tmp_size = tmp_path.stat().st_size
        tmp_path.replace(ckpt_path)
        return True, before, tmp_size
    except Exception:
        return False, before, before


def _yolo_load_labelmap_impl(labelmap_path: Path) -> List[str]:
    try:
        return [line.strip() for line in labelmap_path.read_text().splitlines() if line.strip()]
    except Exception:
        return []


def _yolo_load_run_labelmap_impl(
    run_dir: Path,
    *,
    yolo_load_labelmap_fn: Callable[[Path], List[str]],
    yaml_load_fn: Callable[[str], Any],
) -> List[str]:
    labelmap_path = run_dir / "labelmap.txt"
    labels = yolo_load_labelmap_fn(labelmap_path)
    if labels:
        return labels
    data_yaml = run_dir / "data.yaml"
    if data_yaml.exists():
        try:
            payload = yaml_load_fn(data_yaml.read_text())
            names = payload.get("names") if isinstance(payload, dict) else None
            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys())]
            if isinstance(names, list):
                return [str(x) for x in names]
        except Exception:
            pass
    return []


def _rfdetr_load_labelmap_impl(
    dataset_root: Path,
    coco_train_json: Optional[str],
    *,
    yolo_load_labelmap_fn: Callable[[Path], List[str]],
    json_load_fn: Callable[[str], Any],
) -> List[str]:
    labelmap_path = dataset_root / "labelmap.txt"
    if labelmap_path.exists():
        return yolo_load_labelmap_fn(labelmap_path)
    coco_path = Path(coco_train_json) if coco_train_json else None
    if coco_path and coco_path.exists():
        try:
            data = json_load_fn(coco_path.read_text())
            categories = data.get("categories", [])
            categories = [c for c in categories if isinstance(c, dict) and "id" in c and "name" in c]
            categories.sort(key=lambda c: int(c.get("id", 0)))
            return [str(c["name"]) for c in categories]
        except Exception:
            return []
    return []


def _collect_yolo_artifacts_impl(run_dir: Path, *, meta_name: str) -> Dict[str, bool]:
    return {
        "best_pt": (run_dir / "best.pt").exists(),
        "metrics_json": (run_dir / "metrics.json").exists(),
        "metrics_series": (run_dir / "metrics_series.json").exists(),
        "results_csv": (run_dir / "results.csv").exists(),
        "args_yaml": (run_dir / "args.yaml").exists(),
        "labelmap": (run_dir / "labelmap.txt").exists(),
        "run_meta": (run_dir / meta_name).exists(),
    }


def _collect_rfdetr_artifacts_impl(run_dir: Path, *, meta_name: str) -> Dict[str, bool]:
    return {
        "best_regular": (run_dir / "checkpoint_best_regular.pth").exists(),
        "best_ema": (run_dir / "checkpoint_best_ema.pth").exists(),
        "best_total": (run_dir / "checkpoint_best_total.pth").exists(),
        "best_optimized": (run_dir / "checkpoint_best_optimized.pt").exists(),
        "results_json": (run_dir / "results.json").exists(),
        "metrics_series": (run_dir / "metrics_series.json").exists(),
        "log_txt": (run_dir / "log.txt").exists(),
        "labelmap": (run_dir / "labelmap.txt").exists(),
        "run_meta": (run_dir / meta_name).exists(),
    }


def _yolo_extract_detections_impl(
    results: Any,
    labelmap: List[str],
    offset_x: float,
    offset_y: float,
    full_w: int,
    full_h: int,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not results:
        return detections
    det_boxes = results[0].boxes if results else None
    if det_boxes is None or det_boxes.xyxy is None:
        return detections
    xyxy = det_boxes.xyxy.cpu().numpy()
    confs = det_boxes.conf.cpu().numpy() if det_boxes.conf is not None else None
    classes = det_boxes.cls.cpu().numpy() if det_boxes.cls is not None else None
    for idx, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        abs_x = max(0.0, min(full_w, x1 + offset_x))
        abs_y = max(0.0, min(full_h, y1 + offset_y))
        abs_w = max(0.0, min(full_w - abs_x, (x2 - x1)))
        abs_h = max(0.0, min(full_h - abs_y, (y2 - y1)))
        class_id = int(classes[idx]) if classes is not None else -1
        class_name = labelmap[class_id] if 0 <= class_id < len(labelmap) else None
        score = float(confs[idx]) if confs is not None else None
        detections.append(
            {
                "bbox": [abs_x, abs_y, abs_w, abs_h],
                "class_id": class_id,
                "class_name": class_name,
                "score": score,
            }
        )
    return detections


def _rfdetr_extract_detections_impl(
    results: Any,
    labelmap: List[str],
    offset_x: float,
    offset_y: float,
    full_w: int,
    full_h: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    detections: List[Dict[str, Any]] = []
    labelmap_shifted = False
    if results is None:
        return detections, labelmap_shifted
    xyxy = getattr(results, "xyxy", None)
    scores = getattr(results, "confidence", None)
    class_ids = getattr(results, "class_id", None)
    if xyxy is None or not len(xyxy):
        return detections, labelmap_shifted
    for idx, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        abs_x = max(0.0, min(full_w, x1 + offset_x))
        abs_y = max(0.0, min(full_h, y1 + offset_y))
        abs_w = max(0.0, min(full_w - abs_x, (x2 - x1)))
        abs_h = max(0.0, min(full_h - abs_y, (y2 - y1)))
        class_id = int(class_ids[idx]) if class_ids is not None else -1
        if labelmap and class_id >= len(labelmap) and 0 <= class_id - 1 < len(labelmap):
            class_id -= 1
            labelmap_shifted = True
        class_name = labelmap[class_id] if 0 <= class_id < len(labelmap) else None
        score = float(scores[idx]) if scores is not None else None
        detections.append(
            {
                "bbox": [abs_x, abs_y, abs_w, abs_h],
                "class_id": class_id,
                "class_name": class_name,
                "score": score,
            }
        )
    return detections, labelmap_shifted


def _resolve_detector_image_impl(
    image_base64: Optional[str],
    image_token: Optional[str],
    *,
    fetch_preloaded_fn: Callable[[str, str], Optional[Any]],
    decode_image_fn: Callable[[Optional[str]], Tuple[Any, Any]],
    store_preloaded_fn: Callable[[str, Any, str], None],
    hash_fn: Callable[[bytes], str],
) -> Tuple[Any, Any, str]:
    if image_token:
        for variant in ("sam1", "sam3"):
            cached = fetch_preloaded_fn(image_token, variant)
            if cached is not None:
                pil_img = __import__("PIL").Image.fromarray(cached)
                return pil_img, cached, image_token
        if image_base64:
            pil_img, np_img = decode_image_fn(image_base64)
            token = hash_fn(np_img.tobytes())
            store_preloaded_fn(token, np_img, "sam1")
            return pil_img, np_img, token
        raise RuntimeError("image_token_not_found")
    pil_img, np_img = decode_image_fn(image_base64)
    token = hash_fn(np_img.tobytes())
    store_preloaded_fn(token, np_img, "sam1")
    return pil_img, np_img, token


def _flatten_metrics_impl(obj: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_metrics_impl(value, next_prefix, out)
    else:
        out[prefix] = obj
    return out


def _lookup_metric_impl(flat: Dict[str, Any], keys: List[str]) -> Optional[float]:
    if not flat:
        return None
    lowered = {str(k).lower(): v for k, v in flat.items()}
    for key in keys:
        if key in flat:
            value = flat[key]
        else:
            value = lowered.get(key.lower())
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _yolo_metrics_summary_impl(run_dir: Path, *, read_csv_last_row_fn: Callable[[Path], Optional[Dict[str, str]]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            flat = _flatten_metrics_impl(data)
            summary["map50_95"] = _lookup_metric_impl(flat, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map50-95"])
            summary["map50"] = _lookup_metric_impl(flat, ["metrics/mAP50(B)", "metrics/mAP50", "map50"])
            summary["precision"] = _lookup_metric_impl(flat, ["metrics/precision(B)", "metrics/precision", "precision"])
            summary["recall"] = _lookup_metric_impl(flat, ["metrics/recall(B)", "metrics/recall", "recall"])
        except Exception:
            pass
    if any(value is not None for value in summary.values()):
        return {k: v for k, v in summary.items() if v is not None}
    csv_path = run_dir / "results.csv"
    last_row = read_csv_last_row_fn(csv_path)
    if not last_row:
        return {}

    def _csv_value(name_variants: List[str]) -> Optional[float]:
        for key in name_variants:
            if key in last_row:
                try:
                    return float(last_row[key])
                except Exception:
                    return None
            for col, val in last_row.items():
                if col.strip().lower() == key.strip().lower():
                    try:
                        return float(val)
                    except Exception:
                        return None
        return None

    return {
        "map50_95": _csv_value(["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
        "map50": _csv_value(["metrics/mAP50(B)", "metrics/mAP50"]),
        "precision": _csv_value(["metrics/precision(B)", "metrics/precision"]),
        "recall": _csv_value(["metrics/recall(B)", "metrics/recall"]),
    }


def _rfdetr_metrics_summary_impl(run_dir: Path) -> Dict[str, float]:
    metrics_path = run_dir / "results.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text())
    except Exception:
        return {}
    flat = _flatten_metrics_impl(data)
    return {
        "map": _lookup_metric_impl(flat, ["coco/bbox_mAP", "bbox_mAP", "metrics/bbox_mAP", "map"]),
        "map50": _lookup_metric_impl(flat, ["coco/bbox_mAP50", "bbox_mAP50", "metrics/bbox_mAP50", "map50"]),
        "map75": _lookup_metric_impl(flat, ["coco/bbox_mAP75", "bbox_mAP75", "metrics/bbox_mAP75", "map75"]),
    }


def _clean_metric_summary_impl(summary: Dict[str, Optional[float]]) -> Dict[str, float]:
    return {key: float(value) for key, value in summary.items() if value is not None}


def _list_yolo_runs_impl(
    *,
    job_root: Path,
    dataset_cache_root: Path,
    active_payload: Dict[str, Any],
    load_meta_fn: Callable[[Path], Dict[str, Any]],
    collect_artifacts_fn: Callable[[Path], Dict[str, bool]],
    meta_name: str,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    active_id = active_payload.get("run_id") if isinstance(active_payload, dict) else None
    for entry in job_root.iterdir():
        if not entry.is_dir():
            continue
        if entry == dataset_cache_root:
            continue
        meta_path = entry / meta_name
        if not meta_path.exists():
            continue
        meta = load_meta_fn(entry)
        run_id = meta.get("job_id") or entry.name
        config = meta.get("config") or {}
        dataset = config.get("dataset") or {}
        run_name = config.get("run_name") or dataset.get("label") or dataset.get("id") or run_id
        created_at = meta.get("created_at")
        if not created_at:
            try:
                created_at = entry.stat().st_mtime
            except Exception:
                created_at = None
        runs.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "status": meta.get("status"),
                "message": meta.get("message"),
                "created_at": created_at,
                "updated_at": meta.get("updated_at"),
                "dataset_id": dataset.get("id") or dataset.get("dataset_id"),
                "dataset_label": dataset.get("label"),
                "artifacts": collect_artifacts_fn(entry),
                "is_active": bool(active_id and run_id == active_id),
            }
        )
    runs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return runs


def _list_rfdetr_runs_impl(
    *,
    job_root: Path,
    active_payload: Dict[str, Any],
    load_meta_fn: Callable[[Path], Dict[str, Any]],
    collect_artifacts_fn: Callable[[Path], Dict[str, bool]],
    meta_name: str,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    active_id = active_payload.get("run_id") if isinstance(active_payload, dict) else None
    for entry in job_root.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / meta_name
        if not meta_path.exists():
            continue
        meta = load_meta_fn(entry)
        run_id = meta.get("job_id") or entry.name
        config = meta.get("config") or {}
        dataset = config.get("dataset") or {}
        run_name = config.get("run_name") or dataset.get("label") or dataset.get("id") or run_id
        created_at = meta.get("created_at")
        if not created_at:
            try:
                created_at = entry.stat().st_mtime
            except Exception:
                created_at = None
        runs.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "status": meta.get("status"),
                "message": meta.get("message"),
                "created_at": created_at,
                "updated_at": meta.get("updated_at"),
                "dataset_id": dataset.get("id") or dataset.get("dataset_id"),
                "dataset_label": dataset.get("label"),
                "artifacts": collect_artifacts_fn(entry),
                "is_active": bool(active_id and run_id == active_id),
            }
        )
    runs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return runs


def _agent_tool_run_detector_impl(
    *,
    image_base64: Optional[str],
    image_token: Optional[str],
    detector_id: Optional[str],
    mode: Optional[str],
    conf: Optional[float],
    sahi: Optional[Dict[str, Any]],
    window: Optional[Any],
    window_bbox_2d: Optional[Sequence[float]],
    grid_cell: Optional[str],
    max_det: Optional[int],
    iou: Optional[float],
    merge_iou: Optional[float],
    expected_labelmap: Optional[Sequence[str]],
    register: Optional[bool],
    resolve_image_fn: Callable[[Optional[str], Optional[str], Optional[str]], Tuple[Any, Any, str]],
    normalize_window_fn: Callable[[Optional[Any], int, int], Optional[Tuple[float, float, float, float]]],
    ensure_yolo_runtime_fn: Callable[[], Tuple[Any, List[str], str]],
    ensure_rfdetr_runtime_fn: Callable[[], Tuple[Any, List[str], str]],
    raise_labelmap_mismatch_fn: Callable[[Optional[Sequence[str]], Optional[Sequence[str]], str], None],
    clamp_conf_fn: Callable[[float, List[str]], float],
    clamp_iou_fn: Callable[[float, List[str]], float],
    clamp_max_det_fn: Callable[[int, List[str]], int],
    clamp_slice_params_fn: Callable[[int, float, float, int, int, List[str]], Tuple[int, float, float]],
    slice_image_fn: Callable[[Any, int, float], Tuple[List[Any], List[Tuple[int, int]]]],
    yolo_extract_fn: Callable[[Any, List[str], float, float, int, int], List[Dict[str, Any]]],
    rfdetr_extract_fn: Callable[[Any, List[str], float, float, int, int], Tuple[List[Dict[str, Any]], bool]],
    merge_nms_fn: Callable[[List[Dict[str, Any]], float, int], List[Dict[str, Any]]],
    xywh_to_xyxy_fn: Callable[[Sequence[float]], Tuple[float, float, float, float]],
    det_payload_fn: Callable[..., Dict[str, Any]],
    register_detections_fn: Callable[..., Optional[Dict[str, Any]]],
    cluster_summaries_fn: Callable[[Sequence[int], bool], Dict[str, Any]],
    handles_from_cluster_ids_fn: Callable[[Sequence[int]], List[str]],
    cluster_label_counts_fn: Callable[[Sequence[int]], Dict[str, int]],
    agent_labelmap: Optional[Sequence[str]],
    agent_grid: Any,
    yolo_lock: Any,
    rfdetr_lock: Any,
    http_exception_cls: Any,
) -> Dict[str, Any]:
    pil_img, _, _ = resolve_image_fn(image_base64, image_token, None)
    img_w, img_h = pil_img.size
    mode_norm = (mode or "yolo").strip().lower()
    if mode_norm not in {"yolo", "rfdetr"}:
        raise http_exception_cls(status_code=400, detail="agent_detector_mode_invalid")
    window_xyxy = normalize_window_fn(window, img_w, img_h)
    if window_xyxy is None and window_bbox_2d is not None:
        window_xyxy = normalize_window_fn({"bbox_2d": window_bbox_2d}, img_w, img_h)
    crop_img = pil_img
    offset_x = 0.0
    offset_y = 0.0
    if window_xyxy:
        x1, y1, x2, y2 = window_xyxy
        crop_img = pil_img.crop((x1, y1, x2, y2))
        offset_x, offset_y = x1, y1
    detections: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if mode_norm == "yolo":
        model, labelmap, _task = ensure_yolo_runtime_fn()
        expected = list(expected_labelmap or (agent_labelmap or []))
        raise_labelmap_mismatch_fn(expected=expected or None, actual=labelmap, context="yolo")
        conf_val = clamp_conf_fn(float(conf) if conf is not None else 0.25, warnings)
        iou_val = clamp_iou_fn(float(iou) if iou is not None else 0.45, warnings)
        max_det_val = clamp_max_det_fn(int(max_det) if max_det is not None else 300, warnings)
        raw: List[Dict[str, Any]] = []
        if sahi and sahi.get("enabled"):
            try:
                slice_size = int(sahi.get("slice_size") or 640)
                overlap = float(sahi.get("overlap") or 0.2)
                merge_iou_val = float(merge_iou or sahi.get("merge_iou") or 0.5)
                slice_size, overlap, merge_iou_val = clamp_slice_params_fn(
                    slice_size, overlap, merge_iou_val, crop_img.width, crop_img.height, warnings
                )
                slices, starts = slice_image_fn(crop_img, slice_size, overlap)
                for tile, start in zip(slices, starts):
                    tile_offset_x = float(start[0]) + offset_x
                    tile_offset_y = float(start[1]) + offset_y
                    with yolo_lock:
                        results = model.predict(
                            __import__("PIL").Image.fromarray(tile),
                            conf=conf_val,
                            iou=iou_val,
                            max_det=max_det_val,
                            verbose=False,
                        )
                    raw.extend(yolo_extract_fn(results, labelmap, tile_offset_x, tile_offset_y, img_w, img_h))
                raw = merge_nms_fn(raw, merge_iou_val, max_det_val)
            except http_exception_cls as exc:
                if "sahi_unavailable" in str(exc.detail):
                    warnings.append(str(exc.detail))
                else:
                    warnings.append(f"sahi_failed:{exc.detail}")
                raw = []
        if not raw:
            with yolo_lock:
                results = model.predict(
                    crop_img,
                    conf=conf_val,
                    iou=iou_val,
                    max_det=max_det_val,
                    verbose=False,
                )
            raw = yolo_extract_fn(results, labelmap, offset_x, offset_y, img_w, img_h)
        for det in raw:
            x1, y1, x2, y2 = xywh_to_xyxy_fn(det.get("bbox") or [])
            detections.append(
                det_payload_fn(
                    img_w,
                    img_h,
                    (x1, y1, x2, y2),
                    label=det.get("class_name"),
                    class_id=det.get("class_id"),
                    score=det.get("score"),
                    source="yolo",
                    window=window_xyxy,
                )
            )
    else:
        model, labelmap, _task = ensure_rfdetr_runtime_fn()
        expected = list(expected_labelmap or (agent_labelmap or []))
        raise_labelmap_mismatch_fn(expected=expected or None, actual=labelmap, context="rfdetr")
        conf_val = clamp_conf_fn(float(conf) if conf is not None else 0.25, warnings)
        max_det_val = clamp_max_det_fn(int(max_det) if max_det is not None else 300, warnings)
        raw: List[Dict[str, Any]] = []
        if sahi and sahi.get("enabled"):
            try:
                slice_size = int(sahi.get("slice_size") or 640)
                overlap = float(sahi.get("overlap") or 0.2)
                merge_iou_val = float(merge_iou or sahi.get("merge_iou") or 0.5)
                slice_size, overlap, merge_iou_val = clamp_slice_params_fn(
                    slice_size, overlap, merge_iou_val, crop_img.width, crop_img.height, warnings
                )
                slices, starts = slice_image_fn(crop_img, slice_size, overlap)
                for tile, start in zip(slices, starts):
                    tile_offset_x = float(start[0]) + offset_x
                    tile_offset_y = float(start[1]) + offset_y
                    try:
                        with rfdetr_lock:
                            results = model.predict(__import__("PIL").Image.fromarray(tile), threshold=conf_val)
                    except Exception as exc:  # noqa: BLE001
                        raise http_exception_cls(status_code=500, detail=f"rfdetr_predict_failed:{exc}") from exc
                    extracted, shifted = rfdetr_extract_fn(
                        results, labelmap, tile_offset_x, tile_offset_y, img_w, img_h
                    )
                    if shifted:
                        if expected:
                            raise http_exception_cls(
                                status_code=412,
                                detail="detector_labelmap_shifted:rfdetr",
                            )
                        warnings.append("rfdetr_labelmap_shifted")
                    raw.extend(extracted)
                raw = merge_nms_fn(raw, merge_iou_val, max_det_val)
            except http_exception_cls as exc:
                if "sahi_unavailable" in str(exc.detail):
                    warnings.append(str(exc.detail))
                else:
                    warnings.append(f"sahi_failed:{exc.detail}")
                raw = []
        if not raw:
            try:
                with rfdetr_lock:
                    results = model.predict(crop_img, threshold=conf_val)
            except Exception as exc:  # noqa: BLE001
                raise http_exception_cls(status_code=500, detail=f"rfdetr_predict_failed:{exc}") from exc
            raw, shifted = rfdetr_extract_fn(results, labelmap, offset_x, offset_y, img_w, img_h)
            if shifted:
                if expected:
                    raise http_exception_cls(
                        status_code=412,
                        detail="detector_labelmap_shifted:rfdetr",
                    )
                warnings.append("rfdetr_labelmap_shifted")
        raw.sort(key=lambda det: float(det.get("score") or 0.0), reverse=True)
        for det in raw[:max_det_val]:
            x1, y1, x2, y2 = xywh_to_xyxy_fn(det.get("bbox") or [])
            detections.append(
                det_payload_fn(
                    img_w,
                    img_h,
                    (x1, y1, x2, y2),
                    label=det.get("class_name"),
                    class_id=det.get("class_id"),
                    score=det.get("score"),
                    source="rfdetr",
                    window=window_xyxy,
                )
            )
    register_summary: Optional[Dict[str, Any]] = None
    if register:
        register_summary = register_detections_fn(
            detections,
            img_w=img_w,
            img_h=img_h,
            grid=agent_grid,
            labelmap=agent_labelmap or [],
            background=None,
            source_override=None,
            owner_cell=grid_cell,
        )
    new_cluster_ids = register_summary.get("new_cluster_ids") if isinstance(register_summary, dict) else []
    updated_cluster_ids = register_summary.get("updated_cluster_ids") if isinstance(register_summary, dict) else []
    new_summary = cluster_summaries_fn(new_cluster_ids, include_ids=False)
    new_handles = handles_from_cluster_ids_fn(new_cluster_ids or [])
    updated_handles = handles_from_cluster_ids_fn(updated_cluster_ids or [])
    agent_view = {
        "mode": mode_norm,
        "grid_cell": grid_cell,
        "warnings": warnings or None,
        "new_clusters": register_summary.get("new_clusters") if isinstance(register_summary, dict) else 0,
        "new_handles": new_handles,
        "updated_clusters": len(updated_cluster_ids or []),
        "updated_handles": updated_handles,
        "new_items": new_summary.get("items"),
        "new_items_total": new_summary.get("total"),
        "new_items_truncated": new_summary.get("truncated"),
        "label_counts": cluster_label_counts_fn(new_cluster_ids or []),
    }
    return {
        "detections": detections,
        "warnings": warnings or None,
        "register_summary": register_summary,
        "__agent_view__": agent_view,
    }
