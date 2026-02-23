#!/usr/bin/env python
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import localinferenceapi as api

SOURCE_RUNS = [
    "yolo_full",
    "yolo_sahi",
    "rfdetr_full",
    "rfdetr_sahi",
    "sam3_text_full",
    "sam3_text_windowed",
    "sam3_similarity_full",
    "sam3_similarity_windowed",
]
DEFAULT_EMBED_PROJ_DIM = 1024


def _activate_classifier_runtime(classifier_path: Path, dataset_id: str) -> None:
    labelmap_path = Path(f"uploads/clip_dataset_uploads/{dataset_id}_yolo/labelmap.txt")
    if not labelmap_path.exists():
        raise SystemExit(f"labelmap_missing_for_classifier_activation:{labelmap_path}")
    labelmap_target = (api.UPLOAD_ROOT / "labelmaps" / f"{dataset_id}_labelmap.txt").resolve()
    labelmap_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        src_text = labelmap_path.read_text(encoding="utf-8")
        if (not labelmap_target.exists()) or labelmap_target.read_text(encoding="utf-8") != src_text:
            labelmap_target.write_text(src_text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"labelmap_stage_failed:{exc}") from exc
    try:
        request = api.ActiveModelRequest(
            classifier_path=str(classifier_path),
            labelmap_path=str(labelmap_target),
        )
        api.set_active_model(request)
    except Exception as exc:  # noqa: BLE001
        detail = getattr(exc, "detail", None)
        message = str(detail) if detail else str(exc)
        raise SystemExit(f"classifier_runtime_activation_failed:{message}") from exc


def _load_labelmap(dataset_id: str) -> List[str]:
    labelmap_path = Path(f"uploads/clip_dataset_uploads/{dataset_id}_yolo/labelmap.txt")
    if not labelmap_path.exists():
        raise SystemExit(f"Missing labelmap at {labelmap_path}")
    return [line.strip() for line in labelmap_path.read_text().splitlines() if line.strip()]


def _resolve_image_path(dataset_id: str, image_name: str) -> Path:
    dataset_root = Path("uploads/qwen_runs/datasets") / dataset_id
    for split in ("val", "train"):
        candidate = dataset_root / split / image_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image {image_name} in {dataset_root}")


def _yolo_to_xyxy(bbox: Sequence[float], w: int, h: int) -> List[float]:
    cx, cy, bw, bh = bbox
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [x1, y1, x2, y2]


def _load_gt_geometry_priors(
    dataset_id: str,
    image_names: Sequence[str],
    labelmap: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    yolo_root = Path(f"uploads/clip_dataset_uploads/{dataset_id}_yolo")
    values: Dict[str, Dict[str, List[float]]] = {
        str(lbl).strip().lower(): {"log_area": [], "log_aspect": []}
        for lbl in labelmap
    }
    for image_name in image_names:
        image_name = str(image_name or "").strip()
        if not image_name:
            continue
        label_path = yolo_root / "labels" / "val" / f"{Path(image_name).stem}.txt"
        if not label_path.exists():
            label_path = yolo_root / "labels" / "train" / f"{Path(image_name).stem}.txt"
        if not label_path.exists():
            continue
        try:
            img_path = _resolve_image_path(dataset_id, image_name)
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue
        if img_w <= 0 or img_h <= 0:
            continue
        denom = float(img_w) * float(img_h)
        for raw in label_path.read_text().splitlines():
            parts = [p for p in raw.strip().split() if p]
            if len(parts) < 5:
                continue
            try:
                cat_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except (TypeError, ValueError):
                continue
            if cat_id < 0 or cat_id >= len(labelmap):
                continue
            box = _yolo_to_xyxy([cx, cy, bw, bh], img_w, img_h)
            x1, y1, x2, y2 = box
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            if w <= 0.0 or h <= 0.0:
                continue
            area_norm = max((w * h) / max(denom, 1.0), 1e-12)
            aspect = max(w / h, 1e-12)
            label = str(labelmap[cat_id]).strip().lower()
            values[label]["log_area"].append(float(np.log(area_norm)))
            values[label]["log_aspect"].append(float(np.log(aspect)))

    priors: Dict[str, Dict[str, float]] = {}
    for label, payload in values.items():
        area_vals = payload.get("log_area") or []
        aspect_vals = payload.get("log_aspect") or []
        if not area_vals or not aspect_vals:
            continue
        area_arr = np.asarray(area_vals, dtype=np.float64)
        aspect_arr = np.asarray(aspect_vals, dtype=np.float64)
        priors[label] = {
            "area_mu": float(area_arr.mean()),
            "area_sigma": float(max(area_arr.std(), 1e-6)),
            "aspect_mu": float(aspect_arr.mean()),
            "aspect_sigma": float(max(aspect_arr.std(), 1e-6)),
        }
    return priors


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _max_count_overlap(
    target_bbox: Sequence[float],
    dets: Sequence[Dict[str, Any]],
    iou_thr: float,
) -> Tuple[float, int]:
    max_score = 0.0
    count = 0
    for det in dets:
        if _iou(target_bbox, det["bbox"]) >= iou_thr:
            count += 1
            score = float(det.get("score") or 0.0)
            if score > max_score:
                max_score = score
    return max_score, count


def _max_iou_overlap(
    target_bbox: Sequence[float],
    dets: Sequence[Dict[str, Any]],
) -> float:
    best = 0.0
    for det in dets:
        iou_val = _iou(target_bbox, det["bbox"])
        if iou_val > best:
            best = iou_val
    return float(best)


def _canonical_source_run(
    *,
    stage: Optional[str],
    run: Optional[str],
    source: Optional[str],
    detector_run_id: Optional[str],
    has_window: bool,
) -> Optional[str]:
    stage_name = str(stage or "").strip().lower()
    run_name = str(run or "").strip().lower()
    source_name = str(source or "").strip().lower()
    detector_run_name = str(detector_run_id or "").strip().lower()

    if stage_name == "detector":
        if run_name in {"yolo_full", "yolo_sahi", "rfdetr_full", "rfdetr_sahi"}:
            return run_name
        if detector_run_name.endswith(":sahi"):
            if source_name in {"yolo", "rfdetr"}:
                return f"{source_name}_sahi"
        if detector_run_name.endswith(":full"):
            if source_name in {"yolo", "rfdetr"}:
                return f"{source_name}_full"
    if stage_name == "sam3_text":
        return "sam3_text_windowed" if run_name == "windowed" or has_window else "sam3_text_full"
    if stage_name == "sam3_similarity":
        return "sam3_similarity_windowed" if run_name == "windowed" or has_window else "sam3_similarity_full"
    if source_name in {"yolo", "rfdetr"}:
        if detector_run_name.endswith(":sahi") or run_name.endswith("_sahi"):
            return f"{source_name}_sahi"
        return f"{source_name}_full"
    if source_name == "sam3_text":
        return "sam3_text_windowed" if has_window else "sam3_text_full"
    if source_name == "sam3_similarity":
        return "sam3_similarity_windowed" if has_window else "sam3_similarity_full"
    return None


def _index_provenance_atoms(
    provenance: Any,
    *,
    img_w: int,
    img_h: int,
) -> Dict[str, Dict[str, Any]]:
    atoms_by_id: Dict[str, Dict[str, Any]] = {}
    if not isinstance(provenance, dict):
        return atoms_by_id
    atoms = provenance.get("atoms")
    if not isinstance(atoms, list):
        return atoms_by_id
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        atom_id = str(atom.get("atom_id") or "").strip()
        if not atom_id:
            continue
        label = str(atom.get("label") or "").strip().lower()
        score = atom.get("score")
        try:
            score_val = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score_val = 0.0
        bbox = atom.get("bbox_xyxy_px")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            bbox = None
        if bbox is None:
            bbox_2d = atom.get("bbox_2d")
            if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4:
                try:
                    bbox = api._qwen_bbox_to_xyxy(img_w, img_h, bbox_2d)
                except Exception:
                    bbox = None
        source = str(atom.get("source_primary") or atom.get("source") or atom.get("score_source") or "unknown").strip().lower()
        has_window = bool(atom.get("window_bbox_2d") or atom.get("grid_cell"))
        run_name = _canonical_source_run(
            stage=str(atom.get("stage") or ""),
            run=str(atom.get("run") or ""),
            source=source,
            detector_run_id=str(atom.get("source_detector_run_id") or ""),
            has_window=has_window,
        )
        atoms_by_id[atom_id] = {
            "atom_id": atom_id,
            "label": label,
            "score": score_val,
            "bbox": [float(v) for v in bbox[:4]] if bbox is not None else None,
            "source": source or "unknown",
            "run": run_name,
        }
    return atoms_by_id


def _build_feature_names(
    classifier_classes: Sequence[str],
    embed_proj_dim: int,
    image_embed_proj_dim: int,
    labelmap: Sequence[str],
    term_hash_dim: int,
    sources: Sequence[str],
    source_runs: Sequence[str],
) -> List[str]:
    names: List[str] = []
    for cls in classifier_classes:
        names.append(f"clf_prob::{cls}")
    for idx in range(max(0, int(embed_proj_dim))):
        names.append(f"clf_emb_rp::{idx:03d}")
    for cls in classifier_classes:
        names.append(f"img_clf_prob::{cls}")
    if classifier_classes:
        names.extend(
            [
                "img_clf_prob_label",
                "img_clf_prob_delta_label",
                "img_clf_prob_max",
                "img_clf_prob_entropy",
                "img_clf_prob_cosine",
            ]
        )
    for idx in range(max(0, int(image_embed_proj_dim))):
        names.append(f"img_clf_emb_rp::{idx:03d}")
    for label in labelmap:
        names.append(f"cand_label::{label}")
    for label in labelmap:
        names.append(f"sam3_text_max::{label}")
        names.append(f"sam3_text_count::{label}")
        names.append(f"sam3_sim_max::{label}")
        names.append(f"sam3_sim_count::{label}")
    names.extend(
        [
            "cand_raw_score_yolo",
            "cand_raw_score_rfdetr",
            "cand_raw_score_sam3_text",
            "cand_raw_score_sam3_similarity",
            "cand_score_yolo",
            "cand_score_rfdetr",
            "cand_score_sam3_text",
            "cand_score_sam3_similarity",
            "cand_has_yolo",
            "cand_has_rfdetr",
            "cand_has_sam3_text",
            "cand_has_sam3_similarity",
            "det_iou_max_yolo_same_label",
            "det_iou_max_rfdetr_same_label",
            "det_iou_max_detector_same_label",
            "det_iou_max_detector_any_label",
            "support_count_total",
            "support_atom_count",
            "support_atom_same_label_count",
            "support_run_count",
            "support_source_count",
            "support_score_mean",
            "support_score_max",
            "support_score_std",
            "support_iou_mean",
            "support_iou_max",
            "support_source_entropy",
            "support_source_entropy_norm",
            "support_source_max_share",
            "support_detector_share",
            "support_sam_share",
            "support_detector_count",
            "support_sam_count",
            "geom_center_x",
            "geom_center_y",
            "geom_width",
            "geom_height",
            "geom_area",
            "geom_aspect_ratio",
            "geom_prior_area_z",
            "geom_prior_aspect_z",
            "geom_prior_area_tail",
            "geom_prior_aspect_tail",
        ]
    )
    for run_name in source_runs:
        names.append(f"cand_run_max::{run_name}")
        names.append(f"cand_run_count::{run_name}")
    for label in labelmap:
        names.append(f"ctx_label_count::{label}")
    for label in labelmap:
        for source in sources:
            names.append(f"ctx_source_count::{label}::{source}")
            names.append(f"ctx_source_mean::{label}::{source}")
    names.extend(
        [
            "ctx_total_count",
            "ctx_total_area",
            "ctx_avg_area",
            "ctx_avg_aspect_ratio",
            "ctx_neighbor_count_all",
            "ctx_neighbor_count_same",
            "ctx_neighbor_ratio_same",
            "ctx_neighbor_score_mean_same",
        ]
    )
    for idx in range(max(0, int(term_hash_dim))):
        names.append(f"sam3_term_hash_{idx:03d}")
    return names


def _encode_classifier_features(
    crops: Sequence[Image.Image],
    *,
    head: Optional[Dict[str, Any]],
    batch_size: int,
    device_override: Optional[str],
    min_crop_size: int,
    embed_proj_dim: int,
    embed_proj_seed: int,
    embed_l2_normalize: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not crops:
        return [], []
    embed_proj_dim = max(0, int(embed_proj_dim))
    if head is None:
        return (
            [np.zeros((0,), dtype=np.float32) for _ in crops],
            [np.zeros((embed_proj_dim,), dtype=np.float32) for _ in crops],
        )
    out_probs: List[np.ndarray] = []
    out_embed: List[np.ndarray] = []
    projection_matrix: Optional[np.ndarray] = None
    projection_input_dim: Optional[int] = None
    class_count = len(list(head.get("classes") or []))
    for start in range(0, len(crops), batch_size):
        batch = []
        for crop in crops[start : start + batch_size]:
            w, h = crop.size
            target_w = max(int(min_crop_size), int(w))
            target_h = max(int(min_crop_size), int(h))
            if target_w != w or target_h != h:
                crop = crop.resize((target_w, target_h))
            batch.append(crop)
        feats = api._encode_pil_batch_for_head(batch, head=head, device_override=device_override)
        if feats is None:
            for _ in batch:
                out_probs.append(np.zeros((class_count,), dtype=np.float32))
                out_embed.append(np.zeros((embed_proj_dim,), dtype=np.float32))
            continue
        feats_arr = np.asarray(feats, dtype=np.float32)
        proba = api._clip_head_predict_proba(feats_arr, head)
        if proba is None:
            proba_arr = np.zeros((len(batch), class_count), dtype=np.float32)
        else:
            proba_arr = np.asarray(proba, dtype=np.float32)
            if proba_arr.ndim != 2 or proba_arr.shape[0] != len(batch):
                proba_arr = np.zeros((len(batch), class_count), dtype=np.float32)
        if embed_proj_dim > 0 and feats_arr.ndim == 2 and feats_arr.shape[1] > 0:
            proj_input = feats_arr
            if embed_l2_normalize:
                norms = np.linalg.norm(proj_input, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                proj_input = proj_input / norms
            if projection_matrix is None or projection_input_dim != proj_input.shape[1]:
                rng = np.random.default_rng(int(embed_proj_seed))
                projection_matrix = rng.standard_normal(
                    (proj_input.shape[1], embed_proj_dim), dtype=np.float32
                )
                projection_matrix = projection_matrix / np.sqrt(float(max(1, proj_input.shape[1])))
                projection_input_dim = proj_input.shape[1]
            proj_arr = np.asarray(proj_input @ projection_matrix, dtype=np.float32)
        else:
            proj_arr = np.zeros((len(batch), embed_proj_dim), dtype=np.float32)
        if proj_arr.ndim != 2 or proj_arr.shape[0] != len(batch):
            proj_arr = np.zeros((len(batch), embed_proj_dim), dtype=np.float32)
        for idx in range(len(batch)):
            out_probs.append(np.asarray(proba_arr[idx], dtype=np.float32))
            out_embed.append(np.asarray(proj_arr[idx], dtype=np.float32))
    return out_probs, out_embed


def _parse_embed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embed-proj-dim",
        type=int,
        default=DEFAULT_EMBED_PROJ_DIM,
        help="Projected classifier embedding feature dimension (default: 1024).",
    )
    parser.add_argument(
        "--embed-proj-seed",
        type=int,
        default=42,
        help="Seed for deterministic random projection matrix.",
    )
    parser.add_argument(
        "--image-embed-proj-dim",
        type=int,
        default=0,
        help="Projected full-image classifier embedding dimension (0 disables).",
    )
    parser.add_argument(
        "--image-embed-proj-seed",
        type=int,
        default=4242,
        help="Seed for deterministic full-image projection matrix.",
    )
    parser.set_defaults(embed_l2_normalize=True)
    parser.add_argument(
        "--embed-no-l2-normalize",
        dest="embed_l2_normalize",
        action="store_false",
        help="Disable L2 normalization before projection.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ensemble features from prepass JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL with detections.")
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--classifier-id", default=None, help="Classifier path/id.")
    parser.add_argument(
        "--require-classifier",
        action="store_true",
        help="Fail if classifier features cannot be produced (recommended for calibration runs).",
    )
    parser.add_argument("--sam3-iou", type=float, default=0.5, help="(Deprecated) IoU for SAM3 overlap.")
    parser.add_argument("--support-iou", type=float, default=0.5, help="IoU for cross-source overlap/support features.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for classifier encoding.")
    parser.add_argument("--device", default=None, help="Device override for classifier encoding.")
    parser.add_argument("--min-crop-size", type=int, default=4, help="Min crop size for classifier crops.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
    parser.add_argument("--term-hash-dim", type=int, default=256, help="Hash bucket count for SAM3 term features.")
    parser.add_argument("--context-radius", type=float, default=0.075, help="Neighbor radius in normalized coords.")
    _parse_embed_args(parser)
    args = parser.parse_args()

    labelmap = [lbl.strip().lower() for lbl in _load_labelmap(args.dataset) if lbl.strip()]
    sources = ["yolo", "rfdetr", "sam3_text", "sam3_similarity"]
    source_runs = list(SOURCE_RUNS)
    classifier_head: Optional[Dict[str, Any]] = None
    classifier_classes: List[str] = []
    if args.require_classifier and not args.classifier_id:
        raise SystemExit("classifier_id_required_for_feature_build")
    if args.classifier_id:
        class _ClassifierResolveError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(f"{status_code}:{detail}")
                self.status_code = status_code
                self.detail = detail

        classifier_path = api._resolve_agent_clip_classifier_path_impl(
            args.classifier_id,
            allowed_root=(api.UPLOAD_ROOT / "classifiers").resolve(),
            allowed_exts=api.CLASSIFIER_ALLOWED_EXTS,
            path_is_within_root_fn=api._path_is_within_root_impl,
            http_exception_cls=_ClassifierResolveError,
        )
        _activate_classifier_runtime(classifier_path, args.dataset)
        classifier_head = api._load_clip_head_from_classifier(classifier_path)
        classifier_classes = [str(c) for c in list(classifier_head.get("classes") or [])]
        if args.require_classifier and not classifier_classes:
            raise SystemExit("classifier_classes_missing")
    try:
        embed_proj_dim = max(0, int(args.embed_proj_dim))
    except (TypeError, ValueError):
        embed_proj_dim = DEFAULT_EMBED_PROJ_DIM
    try:
        embed_proj_seed = int(args.embed_proj_seed)
    except (TypeError, ValueError):
        embed_proj_seed = 42
    try:
        image_embed_proj_dim = max(0, int(args.image_embed_proj_dim))
    except (TypeError, ValueError):
        image_embed_proj_dim = 0
    try:
        image_embed_proj_seed = int(args.image_embed_proj_seed)
    except (TypeError, ValueError):
        image_embed_proj_seed = 4242
    embed_l2_normalize = bool(args.embed_l2_normalize)
    if classifier_head is None:
        embed_proj_dim = 0
        image_embed_proj_dim = 0
    if args.require_classifier and classifier_head is None:
        raise SystemExit("classifier_head_missing")
    if args.require_classifier and embed_proj_dim <= 0:
        raise SystemExit("embed_proj_dim_must_be_positive")

    records: List[Dict[str, Any]] = []
    input_path = Path(args.input)
    for line in input_path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    if args.max_images is not None:
        records = records[: max(0, int(args.max_images))]
    image_names = sorted({str(rec.get("image") or "").strip() for rec in records if str(rec.get("image") or "").strip()})
    gt_geom_priors = _load_gt_geometry_priors(args.dataset, image_names, labelmap)

    sam3_term_list: List[str] = []
    term_seen = set()
    for rec in records:
        for det in rec.get("detections") or []:
            term = det.get("sam3_prompt_term")
            if not term:
                continue
            term_text = str(term).strip()
            if not term_text or term_text in term_seen:
                continue
            term_seen.add(term_text)
            sam3_term_list.append(term_text)
    sam3_term_list.sort()
    term_hash_dim = max(0, int(args.term_hash_dim))
    feature_names = _build_feature_names(
        classifier_classes,
        embed_proj_dim,
        image_embed_proj_dim,
        labelmap,
        term_hash_dim,
        sources,
        source_runs,
    )

    X_rows: List[np.ndarray] = []
    meta_rows: List[str] = []

    for rec in records:
        image_name = rec.get("image")
        if not image_name:
            continue
        try:
            img_path = _resolve_image_path(args.dataset, image_name)
        except Exception:
            continue
        try:
            with Image.open(img_path) as img:
                pil_img = img.convert("RGB")
        except Exception:
            continue
        img_w, img_h = pil_img.size
        detections = rec.get("detections") or []
        atoms_by_id = _index_provenance_atoms(rec.get("provenance"), img_w=img_w, img_h=img_h)

        candidates: List[Dict[str, Any]] = []
        sam3_text_by_label: Dict[str, List[Dict[str, Any]]] = {}
        sam3_text_by_term: Dict[str, List[Dict[str, Any]]] = {}
        sam3_sim_by_label: Dict[str, List[Dict[str, Any]]] = {}
        yolo_by_label: Dict[str, List[Dict[str, Any]]] = {}
        rfdetr_by_label: Dict[str, List[Dict[str, Any]]] = {}

        for det in detections:
            label = str(det.get("label") or "").strip().lower()
            if not label:
                continue
            bbox = det.get("bbox_xyxy_px")
            if not bbox and det.get("bbox_yolo"):
                bbox = _yolo_to_xyxy(det.get("bbox_yolo"), img_w, img_h)
            if not bbox or len(bbox) < 4:
                continue
            try:
                score = float(det.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            score_source = str(det.get("score_source") or det.get("source") or "").strip().lower() or "unknown"
            term = str(det.get("sam3_prompt_term") or "").strip() or None
            atom_ids = []
            raw_atom_ids = det.get("prepass_atom_ids")
            if isinstance(raw_atom_ids, (list, tuple)):
                for raw_atom_id in raw_atom_ids:
                    atom_id = str(raw_atom_id or "").strip()
                    if atom_id and atom_id not in atom_ids:
                        atom_ids.append(atom_id)
            source_list_raw = det.get("source_list") or []
            source_list = sorted(
                {
                    str(src).strip().lower()
                    for src in source_list_raw
                    if str(src).strip()
                }
            )
            if score_source and score_source not in source_list:
                source_list.append(score_source)
                source_list = sorted(set(source_list))
            raw_score_map = det.get("score_by_source")
            score_by_source: Dict[str, float] = {}
            if isinstance(raw_score_map, dict):
                for raw_src, raw_src_score in raw_score_map.items():
                    src_name = str(raw_src or "").strip().lower()
                    if not src_name:
                        continue
                    try:
                        src_score_val = float(raw_src_score)
                    except (TypeError, ValueError):
                        continue
                    prev = score_by_source.get(src_name)
                    if prev is None or src_score_val > prev:
                        score_by_source[src_name] = src_score_val
            if score_source not in score_by_source:
                score_by_source[score_source] = score
            entry = {"label": label, "bbox": [float(v) for v in bbox[:4]], "score": score, "term": term}
            for src in source_list:
                if src not in sources:
                    continue
                src_entry = {
                    "label": label,
                    "bbox": [float(v) for v in bbox[:4]],
                    "score": float(score_by_source.get(src) or 0.0),
                    "term": term,
                }
                if src == "sam3_text":
                    sam3_text_by_label.setdefault(label, []).append(src_entry)
                    if term:
                        sam3_text_by_term.setdefault(term, []).append(src_entry)
                elif src == "sam3_similarity":
                    sam3_sim_by_label.setdefault(label, []).append(src_entry)
                elif src == "yolo":
                    yolo_by_label.setdefault(label, []).append(src_entry)
                elif src == "rfdetr":
                    rfdetr_by_label.setdefault(label, []).append(src_entry)
            x1, y1, x2, y2 = entry["bbox"]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            denom_w = float(img_w) if img_w else 1.0
            denom_h = float(img_h) if img_h else 1.0
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area_norm = (w / denom_w) * (h / denom_h)
            aspect = (w / h) if h > 0.0 else 0.0
            candidates.append(
                {
                    "label": label,
                    "bbox": entry["bbox"],
                    "score": score,
                    "score_source": score_source,
                    "source_list": source_list,
                    "score_by_source": score_by_source,
                    "sam3_prompt_term": term,
                    "prepass_atom_ids": atom_ids,
                    "center_x_norm": cx / denom_w if denom_w else 0.0,
                    "center_y_norm": cy / denom_h if denom_h else 0.0,
                    "area_norm": area_norm,
                    "aspect": aspect,
                }
            )

        # build image context
        label_counts: Dict[str, int] = {lbl: 0 for lbl in labelmap}
        source_counts: Dict[Tuple[str, str], int] = {(lbl, src): 0 for lbl in labelmap for src in sources}
        source_score_sums: Dict[Tuple[str, str], float] = {(lbl, src): 0.0 for lbl in labelmap for src in sources}
        total_count = 0
        total_area = 0.0
        total_aspect = 0.0
        for cand in candidates:
            lbl = cand["label"]
            if lbl in label_counts:
                label_counts[lbl] += 1
            total_count += 1
            total_area += float(cand.get("area_norm") or 0.0)
            total_aspect += float(cand.get("aspect") or 0.0)
            for src in cand.get("source_list") or []:
                key = (lbl, src)
                if key in source_counts:
                    source_counts[key] += 1
            for src, src_score in (cand.get("score_by_source") or {}).items():
                src_name = str(src or "").strip().lower()
                if src_name in sources:
                    source_score_sums[(lbl, src_name)] += float(src_score or 0.0)

        # neighbor index
        radius = max(1e-6, float(args.context_radius))
        bin_size = radius
        bins: Dict[Tuple[int, int], List[int]] = {}
        centers = []
        for idx, cand in enumerate(candidates):
            cx = float(cand.get("center_x_norm") or 0.0)
            cy = float(cand.get("center_y_norm") or 0.0)
            centers.append((cx, cy))
            bx = int(cx / bin_size) if bin_size > 0 else 0
            by = int(cy / bin_size) if bin_size > 0 else 0
            bins.setdefault((bx, by), []).append(idx)

        crops = [pil_img.crop(tuple(cand["bbox"])) for cand in candidates]
        proba_rows, embed_rows = _encode_classifier_features(
            crops,
            head=classifier_head,
            batch_size=max(1, int(args.batch_size)),
            device_override=args.device,
            min_crop_size=max(1, int(args.min_crop_size)),
            embed_proj_dim=embed_proj_dim,
            embed_proj_seed=embed_proj_seed,
            embed_l2_normalize=embed_l2_normalize,
        )
        image_proba_rows, image_embed_rows = _encode_classifier_features(
            [pil_img],
            head=classifier_head,
            batch_size=1,
            device_override=args.device,
            min_crop_size=max(1, int(args.min_crop_size)),
            embed_proj_dim=image_embed_proj_dim,
            embed_proj_seed=image_embed_proj_seed,
            embed_l2_normalize=embed_l2_normalize,
        )
        if classifier_head and len(proba_rows) != len(candidates):
            proba_rows = [np.zeros((len(classifier_classes),), dtype=np.float32) for _ in candidates]
        if embed_proj_dim > 0 and len(embed_rows) != len(candidates):
            embed_rows = [np.zeros((embed_proj_dim,), dtype=np.float32) for _ in candidates]
        if classifier_head and len(classifier_classes) > 0:
            if len(image_proba_rows) == 1:
                image_probs = np.asarray(image_proba_rows[0], dtype=np.float32).reshape(-1)
            else:
                image_probs = np.zeros((len(classifier_classes),), dtype=np.float32)
            if image_probs.shape[0] != len(classifier_classes):
                image_probs = np.zeros((len(classifier_classes),), dtype=np.float32)
        else:
            image_probs = np.zeros((0,), dtype=np.float32)
        if image_embed_proj_dim > 0:
            if len(image_embed_rows) == 1:
                image_embed = np.asarray(image_embed_rows[0], dtype=np.float32).reshape(-1)
            else:
                image_embed = np.zeros((image_embed_proj_dim,), dtype=np.float32)
            if image_embed.shape[0] != image_embed_proj_dim:
                image_embed = np.zeros((image_embed_proj_dim,), dtype=np.float32)
        else:
            image_embed = np.zeros((0,), dtype=np.float32)
        classifier_idx_by_label = {
            str(name).strip().lower(): idx for idx, name in enumerate(classifier_classes)
        }
        yolo_all = [det for dets in yolo_by_label.values() for det in dets]
        rfdetr_all = [det for dets in rfdetr_by_label.values() for det in dets]
        detector_all = yolo_all + rfdetr_all

        for idx, cand in enumerate(candidates):
            label = cand["label"]
            bbox = cand["bbox"]
            feat: List[float] = []
            cand_probs = np.zeros((len(classifier_classes),), dtype=np.float32)
            if classifier_classes:
                cand_probs = np.asarray(proba_rows[idx], dtype=np.float32).reshape(-1)
                if cand_probs.shape[0] != len(classifier_classes):
                    cand_probs = np.zeros((len(classifier_classes),), dtype=np.float32)
                feat.extend([float(v) for v in cand_probs])
            if embed_proj_dim > 0:
                feat.extend([float(v) for v in embed_rows[idx]])
            if classifier_classes:
                feat.extend([float(v) for v in image_probs])
                cls_idx = classifier_idx_by_label.get(label)
                cand_label_prob = float(cand_probs[cls_idx]) if cls_idx is not None else 0.0
                image_label_prob = float(image_probs[cls_idx]) if cls_idx is not None else 0.0
                image_prob_max = float(np.max(image_probs)) if image_probs.size else 0.0
                image_prob_entropy = 0.0
                if image_probs.size:
                    clipped = np.clip(image_probs.astype(np.float64), 1e-12, 1.0)
                    image_prob_entropy = float(-np.sum(clipped * np.log(clipped)))
                prob_cosine = 0.0
                if cand_probs.size and image_probs.size:
                    cand_norm = float(np.linalg.norm(cand_probs))
                    image_norm = float(np.linalg.norm(image_probs))
                    if cand_norm > 0.0 and image_norm > 0.0:
                        prob_cosine = float(np.dot(cand_probs, image_probs) / (cand_norm * image_norm))
                feat.extend(
                    [
                        image_label_prob,
                        cand_label_prob - image_label_prob,
                        image_prob_max,
                        image_prob_entropy,
                        prob_cosine,
                    ]
                )
            if image_embed_proj_dim > 0:
                feat.extend([float(v) for v in image_embed])
            for class_name in labelmap:
                feat.append(1.0 if label == class_name else 0.0)

            for class_name in labelmap:
                text_list = sam3_text_by_label.get(class_name, [])
                sim_list = sam3_sim_by_label.get(class_name, [])
                text_max, text_count = _max_count_overlap(bbox, text_list, args.support_iou)
                sim_max, sim_count = _max_count_overlap(bbox, sim_list, args.support_iou)
                feat.extend([text_max, float(text_count), sim_max, float(sim_count)])

            score_yolo, count_yolo = _max_count_overlap(bbox, yolo_by_label.get(label, []), args.support_iou)
            score_rfdetr, count_rfdetr = _max_count_overlap(bbox, rfdetr_by_label.get(label, []), args.support_iou)
            score_sam3_text, count_sam3_text = _max_count_overlap(bbox, sam3_text_by_label.get(label, []), args.support_iou)
            score_sam3_sim, count_sam3_sim = _max_count_overlap(bbox, sam3_sim_by_label.get(label, []), args.support_iou)
            cand_score_map = cand.get("score_by_source") or {}
            raw_score_yolo = float(cand_score_map.get("yolo") or 0.0)
            raw_score_rfdetr = float(cand_score_map.get("rfdetr") or 0.0)
            raw_score_sam3_text = float(cand_score_map.get("sam3_text") or 0.0)
            raw_score_sam3_sim = float(cand_score_map.get("sam3_similarity") or 0.0)
            has_yolo = 1.0 if score_yolo > 0.0 or count_yolo > 0 else 0.0
            has_rfdetr = 1.0 if score_rfdetr > 0.0 or count_rfdetr > 0 else 0.0
            has_sam3_text = 1.0 if score_sam3_text > 0.0 or count_sam3_text > 0 else 0.0
            has_sam3_sim = 1.0 if score_sam3_sim > 0.0 or count_sam3_sim > 0 else 0.0
            det_iou_yolo_same = _max_iou_overlap(bbox, yolo_by_label.get(label, []))
            det_iou_rfdetr_same = _max_iou_overlap(bbox, rfdetr_by_label.get(label, []))
            det_iou_same = max(det_iou_yolo_same, det_iou_rfdetr_same)
            det_iou_any = _max_iou_overlap(bbox, detector_all)
            support_total = float(count_yolo + count_rfdetr + count_sam3_text + count_sam3_sim)
            support_atom_ids = [atom_id for atom_id in (cand.get("prepass_atom_ids") or []) if atom_id in atoms_by_id]
            support_atoms = [atoms_by_id[atom_id] for atom_id in support_atom_ids]
            support_atoms_same = [atom for atom in support_atoms if atom.get("label") == label]
            if not support_atoms_same:
                support_atoms_same = support_atoms
            run_count_map = {run_name: 0 for run_name in source_runs}
            run_max_map = {run_name: 0.0 for run_name in source_runs}
            support_scores: List[float] = []
            support_sources: List[str] = []
            support_source_counts: Dict[str, int] = {}
            support_iou_vals: List[float] = []
            support_runs: List[str] = []
            for atom in support_atoms_same:
                run_name = atom.get("run")
                score_val = float(atom.get("score") or 0.0)
                if run_name in run_count_map:
                    run_count_map[run_name] += 1
                    if score_val > run_max_map[run_name]:
                        run_max_map[run_name] = score_val
                    if run_name not in support_runs:
                        support_runs.append(run_name)
                source_name = str(atom.get("source") or "").strip().lower()
                if source_name and source_name not in support_sources:
                    support_sources.append(source_name)
                if source_name:
                    support_source_counts[source_name] = support_source_counts.get(source_name, 0) + 1
                support_scores.append(score_val)
                atom_bbox = atom.get("bbox")
                if isinstance(atom_bbox, (list, tuple)) and len(atom_bbox) >= 4:
                    support_iou_vals.append(_iou(bbox, atom_bbox))
            support_score_mean = float(np.mean(support_scores)) if support_scores else 0.0
            support_score_max = float(np.max(support_scores)) if support_scores else 0.0
            support_score_std = float(np.std(support_scores)) if support_scores else 0.0
            support_iou_mean = float(np.mean(support_iou_vals)) if support_iou_vals else 0.0
            support_iou_max = float(np.max(support_iou_vals)) if support_iou_vals else 0.0
            support_total_count = int(sum(int(v) for v in support_source_counts.values()))
            support_source_entropy = 0.0
            support_source_entropy_norm = 0.0
            support_source_max_share = 0.0
            support_detector_count = int(
                support_source_counts.get("yolo", 0) + support_source_counts.get("rfdetr", 0)
            )
            support_sam_count = int(
                support_source_counts.get("sam3_text", 0) + support_source_counts.get("sam3_similarity", 0)
            )
            support_detector_share = 0.0
            support_sam_share = 0.0
            if support_total_count > 0:
                probs = [float(v) / float(support_total_count) for v in support_source_counts.values() if int(v) > 0]
                if probs:
                    support_source_entropy = float(-sum(p * np.log(max(p, 1e-12)) for p in probs))
                    support_source_max_share = float(max(probs))
                    if len(probs) > 1:
                        support_source_entropy_norm = float(
                            support_source_entropy / np.log(float(len(probs)))
                        )
                support_detector_share = float(support_detector_count) / float(support_total_count)
                support_sam_share = float(support_sam_count) / float(support_total_count)
            feat.extend(
                [
                    raw_score_yolo,
                    raw_score_rfdetr,
                    raw_score_sam3_text,
                    raw_score_sam3_sim,
                    score_yolo,
                    score_rfdetr,
                    score_sam3_text,
                    score_sam3_sim,
                    has_yolo,
                    has_rfdetr,
                    has_sam3_text,
                    has_sam3_sim,
                    det_iou_yolo_same,
                    det_iou_rfdetr_same,
                    det_iou_same,
                    det_iou_any,
                    support_total,
                    float(len(support_atoms)),
                    float(len(support_atoms_same)),
                    float(len(support_runs)),
                    float(len(support_sources)),
                    support_score_mean,
                    support_score_max,
                    support_score_std,
                    support_iou_mean,
                    support_iou_max,
                    support_source_entropy,
                    support_source_entropy_norm,
                    support_source_max_share,
                    support_detector_share,
                    support_sam_share,
                    float(support_detector_count),
                    float(support_sam_count),
                ]
            )
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            denom_w = float(img_w) if img_w else 0.0
            denom_h = float(img_h) if img_h else 0.0
            denom_area = denom_w * denom_h
            area_norm = (w * h / denom_area) if denom_area else 0.0
            aspect = (w / h) if h > 0.0 else 0.0
            geom_prior = gt_geom_priors.get(label) or {}
            if geom_prior and area_norm > 0.0 and aspect > 0.0:
                area_log = float(np.log(max(area_norm, 1e-12)))
                aspect_log = float(np.log(max(aspect, 1e-12)))
                area_sigma = float(max(float(geom_prior.get("area_sigma", 1.0)), 1e-6))
                aspect_sigma = float(max(float(geom_prior.get("aspect_sigma", 1.0)), 1e-6))
                area_z = abs((area_log - float(geom_prior.get("area_mu", area_log))) / area_sigma)
                aspect_z = abs((aspect_log - float(geom_prior.get("aspect_mu", aspect_log))) / aspect_sigma)
            else:
                area_z = 0.0
                aspect_z = 0.0
            area_tail = max(0.0, area_z - 2.0)
            aspect_tail = max(0.0, aspect_z - 2.0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            feat.extend(
                [
                    (cx / denom_w) if denom_w else 0.0,
                    (cy / denom_h) if denom_h else 0.0,
                    (w / denom_w) if denom_w else 0.0,
                    (h / denom_h) if denom_h else 0.0,
                    area_norm,
                    aspect,
                    area_z,
                    aspect_z,
                    area_tail,
                    aspect_tail,
                ]
            )
            for run_name in source_runs:
                feat.append(float(run_max_map.get(run_name) or 0.0))
                feat.append(float(run_count_map.get(run_name) or 0.0))

            # context features (subtract candidate contribution)
            ctx_label_count = label_counts.get(label, 0) - 1
            for class_name in labelmap:
                feat.append(float(ctx_label_count if class_name == label else label_counts.get(class_name, 0)))
            for class_name in labelmap:
                for src in sources:
                    base_count = source_counts.get((class_name, src), 0)
                    if class_name == label and src in (cand.get("source_list") or []):
                        base_count = max(0, base_count - 1)
                    base_sum = source_score_sums.get((class_name, src), 0.0)
                    if class_name == label and src in cand_score_map:
                        base_sum -= float(cand_score_map.get(src) or 0.0)
                    mean_val = base_sum / base_count if base_count > 0 else 0.0
                    feat.append(float(base_count))
                    feat.append(float(mean_val))

            ctx_total_count = max(0, total_count - 1)
            ctx_total_area = max(0.0, total_area - float(cand.get("area_norm") or 0.0))
            ctx_total_aspect = max(0.0, total_aspect - float(cand.get("aspect") or 0.0))
            ctx_avg_area = ctx_total_area / ctx_total_count if ctx_total_count > 0 else 0.0
            ctx_avg_aspect = ctx_total_aspect / ctx_total_count if ctx_total_count > 0 else 0.0

            # neighbor features
            cxn, cyn = centers[idx]
            bx = int(cxn / bin_size) if bin_size > 0 else 0
            by = int(cyn / bin_size) if bin_size > 0 else 0
            neighbor_all = 0
            neighbor_same = 0
            neighbor_score_sum = 0.0
            r2 = radius * radius
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in bins.get((bx + dx, by + dy), []):
                        if j == idx:
                            continue
                        nx, ny = centers[j]
                        ddx = cxn - nx
                        ddy = cyn - ny
                        if ddx * ddx + ddy * ddy <= r2:
                            neighbor_all += 1
                            if candidates[j]["label"] == label:
                                neighbor_same += 1
                                neighbor_score_sum += float(candidates[j].get("score") or 0.0)
            neighbor_ratio = (neighbor_same / neighbor_all) if neighbor_all > 0 else 0.0
            neighbor_score_mean = (neighbor_score_sum / neighbor_same) if neighbor_same > 0 else 0.0
            feat.extend(
                [
                    float(ctx_total_count),
                    float(ctx_total_area),
                    float(ctx_avg_area),
                    float(ctx_avg_aspect),
                    float(neighbor_all),
                    float(neighbor_same),
                    float(neighbor_ratio),
                    float(neighbor_score_mean),
                ]
            )
            if term_hash_dim > 0:
                term_features = [0.0 for _ in range(term_hash_dim)]
                for term, term_dets in sam3_text_by_term.items():
                    _, term_count = _max_count_overlap(bbox, term_dets, args.support_iou)
                    if term_count <= 0:
                        continue
                    digest = hashlib.md5(term.encode("utf-8")).hexdigest()
                    bucket = int(digest, 16) % term_hash_dim
                    term_features[bucket] += float(term_count)
                feat.extend(term_features)

            X_rows.append(np.asarray(feat, dtype=np.float32))
            meta_rows.append(
                json.dumps(
                    {
                        "image": image_name,
                        "label": label,
                        "bbox_xyxy_px": bbox,
                        "score": cand.get("score"),
                        "score_source": cand.get("score_source"),
                        "source_list": list(cand.get("source_list") or []),
                        "score_by_source": dict(cand.get("score_by_source") or {}),
                        "sam3_prompt_term": cand.get("sam3_prompt_term"),
                        "prepass_atom_ids": list(cand.get("prepass_atom_ids") or []),
                    },
                    ensure_ascii=True,
                )
            )

    X = np.stack(X_rows, axis=0) if X_rows else np.zeros((0, len(feature_names)), dtype=np.float32)
    meta = np.asarray(meta_rows, dtype=object)
    if args.require_classifier:
        embed_idx = [
            idx
            for idx, name in enumerate(feature_names)
            if name.startswith("clf_emb_rp::") or name.startswith("embed_proj_")
        ]
        if not embed_idx:
            raise SystemExit("embed_features_missing")
        clf_prob_idx = [idx for idx, name in enumerate(feature_names) if name.startswith("clf_prob::")]
        if not clf_prob_idx:
            raise SystemExit("classifier_prob_features_missing")
        if X.shape[0] > 0:
            if np.allclose(X[:, embed_idx], 0.0):
                raise SystemExit("embed_features_all_zero")
            if np.allclose(X[:, clf_prob_idx], 0.0):
                raise SystemExit("classifier_prob_features_all_zero")
    np.savez(
        args.output,
        X=X,
        meta=meta,
        feature_names=np.asarray(feature_names, dtype=object),
        labelmap=np.asarray(labelmap, dtype=object),
        classifier_classes=np.asarray(classifier_classes, dtype=object),
        sam3_term_list=np.asarray(sam3_term_list, dtype=object),
        sam3_term_hash_dim=int(term_hash_dim),
        sam3_iou=float(args.sam3_iou),
        support_iou=float(args.support_iou),
        context_radius=float(args.context_radius),
        embed_proj_dim=int(embed_proj_dim),
        embed_proj_seed=int(embed_proj_seed),
        image_embed_proj_dim=int(image_embed_proj_dim),
        image_embed_proj_seed=int(image_embed_proj_seed),
        embed_l2_normalize=bool(embed_l2_normalize),
    )


if __name__ == "__main__":
    main()
