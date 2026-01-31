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


def _build_feature_names(
    classifier_classes: Sequence[str],
    labelmap: Sequence[str],
    term_hash_dim: int,
    sources: Sequence[str],
) -> List[str]:
    names: List[str] = []
    for cls in classifier_classes:
        names.append(f"clf_prob::{cls}")
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
            "support_count_total",
            "geom_center_x",
            "geom_center_y",
            "geom_width",
            "geom_height",
            "geom_area",
            "geom_aspect_ratio",
        ]
    )
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


def _encode_classifier_probs(
    crops: Sequence[Image.Image],
    *,
    head: Optional[Dict[str, Any]],
    batch_size: int,
    device_override: Optional[str],
    min_crop_size: int,
) -> List[np.ndarray]:
    if not crops:
        return []
    if head is None:
        return [np.zeros((0,), dtype=np.float32) for _ in crops]
    out: List[np.ndarray] = []
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
                out.append(np.zeros((len(head.get("classes") or []),), dtype=np.float32))
            continue
        proba = api._clip_head_predict_proba(feats, head)
        if proba is None:
            for _ in batch:
                out.append(np.zeros((len(head.get("classes") or []),), dtype=np.float32))
            continue
        for row in proba:
            out.append(np.asarray(row, dtype=np.float32))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ensemble features from prepass JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL with detections.")
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--classifier-id", default=None, help="Classifier path/id.")
    parser.add_argument("--sam3-iou", type=float, default=0.5, help="(Deprecated) IoU for SAM3 overlap.")
    parser.add_argument("--support-iou", type=float, default=0.5, help="IoU for cross-source overlap/support features.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for classifier encoding.")
    parser.add_argument("--device", default=None, help="Device override for classifier encoding.")
    parser.add_argument("--min-crop-size", type=int, default=4, help="Min crop size for classifier crops.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
    parser.add_argument("--term-hash-dim", type=int, default=256, help="Hash bucket count for SAM3 term features.")
    parser.add_argument("--context-radius", type=float, default=0.075, help="Neighbor radius in normalized coords.")
    args = parser.parse_args()

    labelmap = [lbl.strip().lower() for lbl in _load_labelmap(args.dataset) if lbl.strip()]
    sources = ["yolo", "rfdetr", "sam3_text", "sam3_similarity"]
    classifier_head: Optional[Dict[str, Any]] = None
    classifier_classes: List[str] = []
    if args.classifier_id:
        classifier_path = api._resolve_agent_clip_classifier_path(args.classifier_id)
        classifier_head = api._load_clip_head_from_classifier(classifier_path)
        classifier_classes = [str(c) for c in list(classifier_head.get("classes") or [])]

    records: List[Dict[str, Any]] = []
    input_path = Path(args.input)
    for line in input_path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    if args.max_images is not None:
        records = records[: max(0, int(args.max_images))]

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
    feature_names = _build_feature_names(classifier_classes, labelmap, term_hash_dim, sources)

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
            score = float(det.get("score") or 0.0)
            score_source = str(det.get("score_source") or det.get("source") or "").strip().lower()
            term = str(det.get("sam3_prompt_term") or "").strip() or None
            entry = {"label": label, "bbox": [float(v) for v in bbox[:4]], "score": score, "term": term}
            if score_source == "sam3_text":
                sam3_text_by_label.setdefault(label, []).append(entry)
                if term:
                    sam3_text_by_term.setdefault(term, []).append(entry)
            elif score_source == "sam3_similarity":
                sam3_sim_by_label.setdefault(label, []).append(entry)
            elif score_source == "yolo":
                yolo_by_label.setdefault(label, []).append(entry)
            elif score_source == "rfdetr":
                rfdetr_by_label.setdefault(label, []).append(entry)
            x1, y1, x2, y2 = entry["bbox"]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            denom_w = float(img_w) if img_w else 1.0
            denom_h = float(img_h) if img_h else 1.0
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area_norm = (w / denom_w) * (h / denom_h)
            aspect = (w / h) if h > 0.0 else 0.0
            src_list = det.get("source_list") or []
            src_list = [str(s).strip().lower() for s in src_list if s]
            candidates.append(
                {
                    "label": label,
                    "bbox": entry["bbox"],
                    "score": score,
                    "score_source": score_source,
                    "source_list": src_list,
                    "sam3_prompt_term": term,
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
            score_src = str(cand.get("score_source") or "").strip().lower()
            if score_src in sources:
                source_score_sums[(lbl, score_src)] += float(cand.get("score") or 0.0)

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
        proba_rows = _encode_classifier_probs(
            crops,
            head=classifier_head,
            batch_size=max(1, int(args.batch_size)),
            device_override=args.device,
            min_crop_size=max(1, int(args.min_crop_size)),
        )
        if classifier_head and len(proba_rows) != len(candidates):
            proba_rows = [np.zeros((len(classifier_classes),), dtype=np.float32) for _ in candidates]

        for idx, cand in enumerate(candidates):
            label = cand["label"]
            bbox = cand["bbox"]
            feat: List[float] = []
            if classifier_classes:
                feat.extend([float(v) for v in proba_rows[idx]])
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
            raw_score_yolo = cand["score"] if cand.get("score_source") == "yolo" else 0.0
            raw_score_rfdetr = cand["score"] if cand.get("score_source") == "rfdetr" else 0.0
            raw_score_sam3_text = cand["score"] if cand.get("score_source") == "sam3_text" else 0.0
            raw_score_sam3_sim = cand["score"] if cand.get("score_source") == "sam3_similarity" else 0.0
            has_yolo = 1.0 if score_yolo > 0.0 or count_yolo > 0 else 0.0
            has_rfdetr = 1.0 if score_rfdetr > 0.0 or count_rfdetr > 0 else 0.0
            has_sam3_text = 1.0 if score_sam3_text > 0.0 or count_sam3_text > 0 else 0.0
            has_sam3_sim = 1.0 if score_sam3_sim > 0.0 or count_sam3_sim > 0 else 0.0
            support_total = float(count_yolo + count_rfdetr + count_sam3_text + count_sam3_sim)
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
                    support_total,
                ]
            )
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            denom_w = float(img_w) if img_w else 0.0
            denom_h = float(img_h) if img_h else 0.0
            denom_area = denom_w * denom_h
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            feat.extend(
                [
                    (cx / denom_w) if denom_w else 0.0,
                    (cy / denom_h) if denom_h else 0.0,
                    (w / denom_w) if denom_w else 0.0,
                    (h / denom_h) if denom_h else 0.0,
                    (w * h / denom_area) if denom_area else 0.0,
                    (w / h) if h > 0.0 else 0.0,
                ]
            )

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
                    if class_name == label and src == (cand.get("score_source") or ""):
                        base_sum -= float(cand.get("score") or 0.0)
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
                        "sam3_prompt_term": cand.get("sam3_prompt_term"),
                    },
                    ensure_ascii=True,
                )
            )

    X = np.stack(X_rows, axis=0) if X_rows else np.zeros((0, len(feature_names)), dtype=np.float32)
    meta = np.asarray(meta_rows, dtype=object)
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
    )


if __name__ == "__main__":
    main()
