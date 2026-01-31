from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _expand_midpoints_impl(
    values: List[float],
    *,
    fine_step: float = 0.05,
    clamp: Tuple[float, float] = (0.0, 1.0),
    limit: int = 20,
) -> List[float]:
    """Given a sorted list, add midpoints and small +/- offsets for coarse-to-fine sweeps."""
    if not values:
        return values
    lo, hi = clamp
    base = sorted({v for v in values if lo <= v <= hi})
    extras: List[float] = []
    for a, b in zip(base, base[1:]):
        mid = (a + b) / 2.0
        extras.append(mid)
    if fine_step > 0:
        for v in base:
            extras.append(v + fine_step)
            extras.append(v - fine_step)
    merged = sorted({v for v in [*base, *extras] if lo <= v <= hi})
    if len(merged) > limit:
        merged = merged[:limit]
    return merged


def _build_gt_index_for_class_impl(
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]],
    target_class: int,
    *,
    xywh_to_xyxy_fn: Callable[[Sequence[float]], Tuple[float, float, float, float]],
) -> Tuple[Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]], set[str], Dict[int, int]]:
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = {}
    all_keys: set[str] = set()
    per_image_counts: Dict[int, int] = {}
    for img_id, by_cat in gt_by_image_cat.items():
        boxes = by_cat.get(target_class)
        if not boxes:
            continue
        entries: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for idx, bbox in enumerate(boxes):
            key = f"{img_id}:{idx}"
            entries.append((key, xywh_to_xyxy_fn(bbox)))
            all_keys.add(key)
        gt_index[img_id] = entries
        per_image_counts[img_id] = len(entries)
    return gt_index, all_keys, per_image_counts


def _evaluate_prompt_for_class_impl(
    prompt: str,
    *,
    cat_id: int,
    image_ids: List[int],
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]],
    images: Dict[int, Dict[str, Any]],
    score_threshold: float,
    max_dets: int,
    iou_threshold: float,
    image_cache: Dict[int, Image.Image],
    run_sam3_text_inference_fn: Callable[..., Any],
    yolo_to_xyxy_fn: Callable[[int, int, Sequence[float]], Tuple[float, float, float, float]],
    xywh_to_xyxy_fn: Callable[[Sequence[float]], Tuple[float, float, float, float]],
    iou_fn: Callable[[Tuple[float, float, float, float], Tuple[float, float, float, float]], float],
) -> Dict[str, Any]:
    total_gt = 0
    total_preds = 0
    matches = 0
    det_images = 0
    iou_sum = 0.0
    score_sum = 0.0
    matched_scores = 0
    for img_id in image_ids:
        info = images.get(img_id)
        if not info:
            continue
        path = info.get("path")
        width = info.get("width")
        height = info.get("height")
        if not path or width is None or height is None:
            continue
        gts = [*gt_by_image_cat.get(img_id, {}).get(cat_id, [])]
        gt_boxes = [xywh_to_xyxy_fn(b) for b in gts]
        total_gt += len(gt_boxes)
        if not gt_boxes:
            continue
        try:
            pil_img = image_cache[img_id]
        except KeyError:
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception:
                continue
            image_cache[img_id] = pil_img
        preds = run_sam3_text_inference_fn(
            pil_img,
            prompt,
            threshold=score_threshold,
            mask_threshold=0.0,
            limit=max_dets,
        )
        pred_boxes: List[Tuple[float, float, float, float, Optional[float]]] = []
        for det in preds:
            try:
                x1, y1, x2, y2 = yolo_to_xyxy_fn(pil_img.width, pil_img.height, det.bbox)
                pred_boxes.append((x1, y1, x2, y2, det.score))
            except Exception:
                continue
        if not pred_boxes:
            continue
        pred_boxes.sort(key=lambda b: (b[4] if b[4] is not None else 0.0), reverse=True)
        total_preds += len(pred_boxes)
        gt_used = [False] * len(gt_boxes)
        matched_in_image = 0
        for x1, y1, x2, y2, score in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                if gt_used[idx]:
                    continue
                iou = iou_fn((x1, y1, x2, y2), gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                gt_used[best_idx] = True
                matches += 1
                matched_in_image += 1
                iou_sum += best_iou
                if score is not None:
                    score_sum += score
                    matched_scores += 1
        if matched_in_image > 0:
            det_images += 1
    precision = matches / total_preds if total_preds else 0.0
    recall = matches / total_gt if total_gt else 0.0
    det_rate = det_images / len(image_ids) if image_ids else 0.0
    avg_iou = iou_sum / matches if matches else None
    avg_score = score_sum / matched_scores if matched_scores else None
    f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    overall_score = f1 * (0.5 + 0.5 * det_rate)
    fps = max(0, total_preds - matches)
    return {
        "prompt": prompt,
        "precision": precision,
        "recall": recall,
        "det_rate": det_rate,
        "avg_iou": avg_iou,
        "avg_score": avg_score,
        "score": overall_score,
        "f1": f1,
        "preds": total_preds,
        "matches": matches,
        "gts": total_gt,
        "fps": fps,
    }


def _evaluate_prompt_candidate_impl(
    prompt: str,
    threshold: float,
    *,
    cat_id: int,
    image_ids: List[int],
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
    other_gt_index: Optional[Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]]] = None,
    images: Dict[int, Dict[str, Any]],
    iou_threshold: float,
    max_dets: int,
    image_cache: Dict[int, Image.Image],
    cached_detections: Optional[Dict[int, List[Tuple[float, float, float, float, Optional[float]]]]] = None,
    run_sam3_text_inference_fn: Callable[..., Any],
    yolo_to_xyxy_fn: Callable[[int, int, Sequence[float]], Tuple[float, float, float, float]],
    iou_fn: Callable[[Tuple[float, float, float, float], Tuple[float, float, float, float]], float],
) -> Dict[str, Any]:
    total_gt = sum(len(gt_index.get(img_id, [])) for img_id in image_ids)
    total_preds = 0
    conflicts = 0
    matches = 0
    fps = 0
    det_images = 0
    iou_sum = 0.0
    score_sum = 0.0
    matched_scores = 0
    matched_gt_keys: set[str] = set()
    matches_by_image: Dict[int, Dict[str, Any]] = {}
    for img_id in image_ids:
        info = images.get(img_id)
        if not info:
            continue
        path = info.get("path")
        width = info.get("width")
        height = info.get("height")
        if not path or width is None or height is None:
            continue
        gts = gt_index.get(img_id, [])
        gt_used = [False] * len(gts)
        pred_boxes: List[Tuple[float, float, float, float, Optional[float]]] = []
        if cached_detections is not None:
            pred_boxes = cached_detections.get(img_id, [])
        if not pred_boxes:
            try:
                pil_img = image_cache[img_id]
            except KeyError:
                try:
                    pil_img = Image.open(path).convert("RGB")
                except Exception:
                    continue
                image_cache[img_id] = pil_img
            preds = run_sam3_text_inference_fn(
                pil_img,
                prompt,
                threshold=threshold,
                mask_threshold=0.0,
                limit=max_dets,
            )
            for det in preds:
                try:
                    x1, y1, x2, y2 = yolo_to_xyxy_fn(pil_img.width, pil_img.height, det.bbox)
                    pred_boxes.append((x1, y1, x2, y2, det.score))
                except Exception:
                    continue
        if not pred_boxes:
            continue
        filtered = [b for b in pred_boxes if (b[4] if b[4] is not None else 0.0) >= threshold]
        filtered.sort(key=lambda b: (b[4] if b[4] is not None else 0.0), reverse=True)
        pred_boxes = filtered[:max_dets] if max_dets else filtered
        total_preds += len(pred_boxes)
        matched_in_image = 0
        fp_in_image = 0
        matched_keys: List[str] = []
        for x1, y1, x2, y2, score in pred_boxes:
            if other_gt_index:
                other_hits = other_gt_index.get(img_id, [])
                conflict_found = False
                for _, other_box in other_hits:
                    if iou_fn((x1, y1, x2, y2), other_box) >= iou_threshold:
                        conflicts += 1
                        conflict_found = True
                        break
                if conflict_found:
                    continue
            total_preds += 1
            best_iou = 0.0
            best_idx = -1
            for idx, (_, gt_box) in enumerate(gts):
                if gt_used[idx]:
                    continue
                iou = iou_fn((x1, y1, x2, y2), gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                gt_used[best_idx] = True
                matches += 1
                matched_in_image += 1
                matched_key = gts[best_idx][0]
                matched_keys.append(matched_key)
                matched_gt_keys.add(matched_key)
                iou_sum += best_iou
                if score is not None:
                    score_sum += score
                    matched_scores += 1
            else:
                fp_in_image += 1
        fps += fp_in_image
        if matched_in_image > 0:
            det_images += 1
        if matched_keys or fp_in_image:
            matches_by_image[img_id] = {"matched": matched_keys, "fps": fp_in_image}
    precision = matches / total_preds if total_preds else 0.0
    recall = matches / total_gt if total_gt else 0.0
    det_rate = det_images / len(image_ids) if image_ids else 0.0
    avg_iou = iou_sum / matches if matches else None
    avg_score = score_sum / matched_scores if matched_scores else None
    f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    overall_score = f1 * (0.5 + 0.5 * det_rate)
    return {
        "prompt": prompt,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "det_rate": det_rate,
        "avg_iou": avg_iou,
        "avg_score": avg_score,
        "score": overall_score,
        "f1": f1,
        "preds": total_preds,
        "matches": matches,
        "gts": total_gt,
        "fps": fps,
        "conflicts": conflicts,
        "det_images": det_images,
        "matched_gt_keys": matched_gt_keys,
        "matches_by_image": matches_by_image,
    }


def _collect_prompt_detections_impl(
    prompt: str,
    min_threshold: float,
    *,
    image_ids: List[int],
    images: Dict[int, Dict[str, Any]],
    image_cache: Dict[int, Image.Image],
    max_dets: int,
    run_sam3_text_inference_fn: Callable[..., Any],
    yolo_to_xyxy_fn: Callable[[int, int, Sequence[float]], Tuple[float, float, float, float]],
) -> Dict[int, List[Tuple[float, float, float, float, Optional[float]]]]:
    results: Dict[int, List[Tuple[float, float, float, float, Optional[float]]]] = {}
    for img_id in image_ids:
        info = images.get(img_id)
        if not info:
            continue
        path = info.get("path")
        width = info.get("width")
        height = info.get("height")
        if not path or width is None or height is None:
            continue
        try:
            pil_img = image_cache[img_id]
        except KeyError:
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception:
                continue
            image_cache[img_id] = pil_img
        preds = run_sam3_text_inference_fn(
            pil_img,
            prompt,
            threshold=min_threshold,
            mask_threshold=0.0,
            limit=max_dets,
        )
        boxes: List[Tuple[float, float, float, float, Optional[float]]] = []
        for det in preds:
            try:
                x1, y1, x2, y2 = yolo_to_xyxy_fn(pil_img.width, pil_img.height, det.bbox)
                boxes.append((x1, y1, x2, y2, det.score))
            except Exception:
                continue
        if boxes:
            results[img_id] = boxes
    return results


def _build_prompt_recipe_impl(
    candidates: List[Dict[str, Any]],
    all_gt_keys: set[str],
    per_image_gt: Dict[int, int],
    images: Dict[int, Dict[str, Any]],
    image_ids: List[int],
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Pick the best threshold per prompt (highest matched GTs, then lowest FPs, then higher precision).
    best_by_prompt: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        prompt = cand.get("prompt") or ""
        matched_count = len(cand.get("matched_gt_keys") or [])
        fps = cand.get("fps", 0)
        precision = cand.get("precision", 0.0)
        current = best_by_prompt.get(prompt)
        if current is None:
            best_by_prompt[prompt] = cand
            continue
        curr_matched = len(current.get("matched_gt_keys") or [])
        curr_fps = current.get("fps", 0)
        curr_precision = current.get("precision", 0.0)
        better = (matched_count, -fps, precision) > (curr_matched, -curr_fps, curr_precision)
        if better:
            best_by_prompt[prompt] = cand
    ordered_candidates = list(best_by_prompt.values())

    # Simulate per-image early stop: run prompts only on images with uncovered GTs; negatives always contribute FPs.
    remaining_by_image: Dict[int, set[str]] = {
        img_id: {key for key, _ in entries} for img_id, entries in gt_index.items() if entries
    }
    remaining_total = sum(len(v) for v in remaining_by_image.values())
    active_images = {img_id for img_id, keys in remaining_by_image.items() if keys}
    negative_images = {img_id for img_id in image_ids if per_image_gt.get(img_id, 0) == 0}
    steps: List[Dict[str, Any]] = []
    total_fps = 0
    total_duplicates = 0
    used_prompt_keys: set[Tuple[str, float]] = set()
    while remaining_total > 0 and ordered_candidates:
        best = None
        best_score = (-1, -1, -1, -1)
        for cand in ordered_candidates:
            prompt_key = (cand.get("prompt"), cand.get("threshold"))
            if prompt_key in used_prompt_keys:
                continue
            matches_by_image = cand.get("matches_by_image") or {}
            step_gain = 0
            step_fps = 0
            step_duplicates = 0
            step_matches_total = 0
            step_hits_by_image: Dict[int, Dict[str, Any]] = {}
            for img_id in negative_images:
                img_hits = matches_by_image.get(img_id)
                if not img_hits:
                    continue
                matched_total = len(img_hits.get("matched") or [])
                fps_count = max(0, img_hits.get("fps", 0))
                step_matches_total += matched_total
                step_fps += fps_count
                if matched_total or fps_count:
                    step_hits_by_image[img_id] = {
                        "matched": [],
                        "matched_total": matched_total,
                        "fps": fps_count,
                    }
            for img_id in active_images:
                img_hits = matches_by_image.get(img_id)
                if not img_hits:
                    continue
                matched_list = img_hits.get("matched") or []
                matched_set = set(matched_list)
                unmatched = remaining_by_image.get(img_id, set())
                new_hits = matched_set & unmatched
                matched_total = len(matched_set)
                fps_count = max(0, img_hits.get("fps", 0))
                step_gain += len(new_hits)
                step_duplicates += max(0, matched_total - len(new_hits))
                step_matches_total += matched_total
                step_fps += fps_count
                step_hits_by_image[img_id] = {
                    "matched": list(new_hits),
                    "matched_total": matched_total,
                    "fps": fps_count,
                }
            if step_gain <= 0:
                continue
            zero_fp = step_fps == 0
            score_tuple = (
                1 if zero_fp else 0,
                step_gain,
                -step_fps,
                cand.get("precision", 0.0),
            )
            if score_tuple > best_score:
                best = (cand, step_hits_by_image, step_gain, step_fps, step_duplicates, step_matches_total)
                best_score = score_tuple
        if not best:
            break
        cand, step_hits_by_image, gain, step_fps, duplicate_hits, step_matches_total = best
        prompt_key = (cand.get("prompt"), cand.get("threshold"))
        used_prompt_keys.add(prompt_key)
        for img_id, hit_info in step_hits_by_image.items():
            new_hits = set(hit_info.get("matched") or [])
            if not new_hits:
                continue
            current_unmatched = remaining_by_image.get(img_id)
            if current_unmatched is None:
                continue
            remaining_by_image[img_id] = current_unmatched - new_hits
        active_images = {img_id for img_id, keys in remaining_by_image.items() if keys}
        remaining_total = max(0, remaining_total - gain)
        total_duplicates += duplicate_hits
        total_fps += max(0, step_fps)
        covered_after = len(all_gt_keys) - remaining_total
        cum_coverage = covered_after / len(all_gt_keys) if all_gt_keys else 0.0
        preds_in_step = step_matches_total + step_fps
        seq_precision = step_matches_total / preds_in_step if preds_in_step else 0.0
        steps.append(
            {
                "prompt": cand.get("prompt"),
                "threshold": cand.get("threshold"),
                "gain": gain,
                "matches": step_matches_total,
                "fps": step_fps,
                "precision": seq_precision,
                "recall": cand.get("recall"),
                "det_rate": cand.get("det_rate"),
                "avg_iou": cand.get("avg_iou"),
                "avg_score": cand.get("avg_score"),
                "duplicates": duplicate_hits,
                "covered_after": covered_after,
                "cum_coverage": cum_coverage,
                "cum_fps": total_fps,
                "_matches_by_image": step_hits_by_image,
                "_prompt_precision": cand.get("precision"),
                "similarity_score": cand.get("similarity_score"),
            }
        )
    coverage_rate = (len(all_gt_keys) - remaining_total) / len(all_gt_keys) if all_gt_keys else 0.0
    recipe = {
        "steps": [
            {
                **{k: v for k, v in step.items() if not k.startswith("_")},
                "coverage_after": (step.get("covered_after", 0) / len(all_gt_keys)) if all_gt_keys else 0.0,
            }
            for step in steps
        ],
        "summary": {
            "total_gt": len(all_gt_keys),
            "covered": len(all_gt_keys) - remaining_total,
            "coverage_rate": coverage_rate,
            "fps": total_fps,
            "duplicates": total_duplicates,
        },
    }
    coverage_by_image: List[Dict[str, Any]] = []
    coverage_map: Dict[int, Dict[str, Any]] = {}
    for img_id in image_ids:
        info = images.get(img_id, {})
        remaining_keys = remaining_by_image.get(img_id, set())
        is_positive = per_image_gt.get(img_id, 0) > 0
        entry = {
            "image_id": img_id,
            "file_name": info.get("file_name"),
            "gt": per_image_gt.get(img_id, 0),
            "hits": [],
            "type": "pos" if is_positive else "neg",
            "covered": per_image_gt.get(img_id, 0) == 0 or len(remaining_keys) == 0,
        }
        coverage_map[img_id] = entry
        coverage_by_image.append(entry)
    for idx, step in enumerate(steps):
        matches_by_image = step.get("_matches_by_image") or {}
        for img_id, img_info in matches_by_image.items():
            target = coverage_map.get(img_id)
            if not target:
                continue
            matched_list = img_info.get("matched") or []
            fp_count = img_info.get("fps", 0)
            target["hits"].append(
                {
                    "step": idx,
                    "prompt": step.get("prompt"),
                    "threshold": step.get("threshold"),
                    "matched": len(matched_list),
                    "fps": fp_count,
                }
            )
    return recipe, coverage_by_image


def _score_greedy_eval_summaries_impl(
    summaries: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    total_gt = 0
    matched = 0
    fps = 0
    preds = 0
    for _cid, s in (summaries or {}).items():
        if not isinstance(s, dict):
            continue
        try:
            total_gt += int(s.get("gts") or 0)
        except Exception:
            pass
        try:
            matched += int(s.get("matches") or 0)
        except Exception:
            pass
        try:
            fps += int(s.get("fps") or 0)
        except Exception:
            pass
        try:
            preds += int(s.get("preds") or 0)
        except Exception:
            pass
    precision = matched / max(1, matched + fps)
    recall = matched / max(1, total_gt)
    f1 = 0.0
    if precision + recall > 1e-9:
        f1 = 2.0 * precision * recall / (precision + recall)
    score_key = (matched, -fps, float(precision))
    return {
        "gts": total_gt,
        "matches": matched,
        "fps": fps,
        "preds": preds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "key": score_key,
    }


def _gt_instance_key_impl(img_id: int, gt_idx: int) -> int:
    # Pack (image_id, gt_index) into a single int key for compact set operations.
    return (int(img_id) << 20) | int(gt_idx)


def _build_seed_threshold_sweep_grid_impl(
    *,
    base_seed_threshold: float,
    observed_scores: Optional[Sequence[float]] = None,
    limit: int = 64,
) -> List[float]:
    """
    Build a threshold grid for SAM3 seed-stage threshold sweeps.

    Notes:
    - SAM3 text scores can cluster near 0.0, so we include dense points in [0.0..0.1].
    - This is a *seed* sweep grid (SAM score threshold), not a CLIP-head sweep grid.
    """
    try:
        base = float(base_seed_threshold)
    except Exception:
        base = 0.05
    base = max(0.0, min(1.0, base))
    try:
        limit_i = max(4, int(limit))
    except Exception:
        limit_i = 64

    raw: List[float] = []
    raw.extend([0.0, 0.001, 0.002, 0.005])
    raw.extend([round(i * 0.01, 3) for i in range(0, 11)])  # 0.00..0.10
    raw.extend([round(i * 0.05, 3) for i in range(3, 20)])  # 0.15..0.95
    raw.extend([0.975, 0.99, 1.0])
    raw.append(base)

    if observed_scores:
        cleaned_scores: List[float] = []
        for s in observed_scores:
            try:
                val = float(s)
            except Exception:
                continue
            if not (0.0 <= val <= 1.0):
                continue
            cleaned_scores.append(val)
            if len(cleaned_scores) >= 20_000:
                break
        if cleaned_scores:
            try:
                qs = np.quantile(np.asarray(cleaned_scores, dtype=np.float32), np.linspace(0.0, 1.0, num=21))
                raw.extend([float(x) for x in qs.tolist()])
            except Exception:
                pass

    grid = sorted({float(max(0.0, min(1.0, round(v, 4)))) for v in raw})
    if len(grid) > limit_i:
        # Keep low thresholds dense; truncate the tail.
        grid = grid[:limit_i]
    return grid


def _compute_steps_seed_eval_threshold_impl(payload: Any) -> float:
    """
    Seed-stage prompt eval threshold used for steps-mode mining.

    This can be lower than the user-configured `seed_threshold` so seed-threshold curves can represent
    operating points below the base threshold.
    """
    try:
        base = float(getattr(payload, "seed_threshold", 0.05))
    except Exception:
        base = 0.05
    base = max(0.0, min(1.0, float(base)))

    floor_raw = getattr(payload, "steps_seed_eval_floor", None)
    if floor_raw is None:
        return base
    try:
        floor = float(floor_raw)
    except Exception:
        return base
    floor = max(0.0, min(1.0, float(floor)))
    return float(min(base, floor))


def _compute_steps_seed_eval_max_results_impl(payload: Any) -> int:
    """
    Seed-stage prompt eval max_results override used for steps-mode mining.

    This exists because very low `steps_seed_eval_floor` values can return many detections.
    """
    try:
        base = int(getattr(payload, "max_results", 1000) or 1000)
    except Exception:
        base = 1000
    base = max(1, min(5000, int(base)))

    override_raw = getattr(payload, "steps_seed_eval_max_results", None)
    if override_raw is None:
        return base
    try:
        override = int(override_raw)
    except Exception:
        return base
    override = max(1, min(5000, int(override)))
    return override


def _compute_seed_threshold_curve_impl(
    *,
    gt_best_scores: Mapping[Any, float],
    fp_scores: Sequence[float],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    """
    Compute seed-stage (matches,fps,precision) at each threshold.

    - `gt_best_scores` maps a GT instance key -> best SAM score among detections matching that GT.
    - `fp_scores` contains SAM scores for detections that did not match any GT (false positives).

    Precision here mirrors our existing seed-stage notion:
      precision = matches / (matches + fps)

    (Duplicates are intentionally not modeled in this helper; they can be tracked separately.)
    """
    thr_list: List[float] = []
    for t in thresholds:
        try:
            thr_list.append(float(t))
        except Exception:
            continue
    thr_list = sorted({max(0.0, min(1.0, round(v, 6))) for v in thr_list})
    if not thr_list:
        return []

    gt_vals: List[float] = []
    for _k, v in (gt_best_scores or {}).items():
        try:
            val = float(v)
        except Exception:
            continue
        if 0.0 <= val <= 1.0:
            gt_vals.append(val)
    gt_arr = np.asarray(gt_vals, dtype=np.float32) if gt_vals else np.zeros((0,), dtype=np.float32)
    gt_arr.sort()

    fp_arr: np.ndarray
    if fp_scores is None:
        fp_arr = np.zeros((0,), dtype=np.float32)
    else:
        try:
            fp_arr = np.asarray(fp_scores, dtype=np.float32)
        except Exception:
            fp_arr = np.asarray(list(fp_scores), dtype=np.float32) if fp_scores else np.zeros((0,), dtype=np.float32)
    if fp_arr.size:
        try:
            fp_arr = fp_arr[(fp_arr >= 0.0) & (fp_arr <= 1.0)]
        except Exception:
            fp_arr = fp_arr
    fp_arr.sort()

    total_gt = int(gt_arr.size)
    total_fp = int(fp_arr.size)

    curve: List[Dict[str, Any]] = []
    for thr in thr_list:
        try:
            thr_key_gt = gt_arr.dtype.type(float(thr))
        except Exception:
            thr_key_gt = float(thr)
        try:
            thr_key_fp = fp_arr.dtype.type(float(thr))
        except Exception:
            thr_key_fp = float(thr)
        gt_start = int(np.searchsorted(gt_arr, thr_key_gt, side="left")) if total_gt else 0
        fp_start = int(np.searchsorted(fp_arr, thr_key_fp, side="left")) if total_fp else 0
        matches = total_gt - gt_start
        fps = total_fp - fp_start
        precision = float(matches) / float(max(1, matches + fps))
        curve.append(
            {
                "threshold": float(thr),
                "matches": int(matches),
                "fps": int(fps),
                "precision": float(precision),
            }
        )
    return curve


def _select_seed_threshold_operating_point_impl(
    curve: Sequence[Dict[str, Any]],
    *,
    min_precision: Optional[float] = None,
    max_fps: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select a single operating point from a seed-threshold curve.

    Selection rule:
      1) Prefer points that satisfy constraints (precision >= min_precision, fps <= max_fps).
      2) Among those, maximize matches, then minimize fps, then prefer lower threshold (less brittle).
      3) If none satisfy, fall back to maximizing precision, then matches, then minimizing fps.
    """
    if not curve:
        return None

    try:
        min_prec = float(min_precision) if min_precision is not None else None
    except Exception:
        min_prec = None
    if min_prec is not None:
        min_prec = max(0.0, min(1.0, min_prec))

    try:
        max_fps_i = int(max_fps) if max_fps is not None else None
    except Exception:
        max_fps_i = None
    if max_fps_i is not None:
        max_fps_i = max(0, max_fps_i)

    def _as_point(p: Dict[str, Any]) -> Tuple[float, int, int, float]:
        try:
            thr = float(p.get("threshold") or 0.0)
        except Exception:
            thr = 0.0
        try:
            matches = int(p.get("matches") or 0)
        except Exception:
            matches = 0
        try:
            fps = int(p.get("fps") or 0)
        except Exception:
            fps = 0
        try:
            precision = float(p.get("precision") or 0.0)
        except Exception:
            precision = 0.0
        return thr, matches, fps, precision

    candidates: List[Dict[str, Any]] = []
    for p in curve:
        if not isinstance(p, dict):
            continue
        _thr, _matches, fps, precision = _as_point(p)
        ok = True
        if min_prec is not None and precision < float(min_prec):
            ok = False
        if max_fps_i is not None and fps > int(max_fps_i):
            ok = False
        if ok:
            candidates.append(p)

    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[int, int, float]] = None
    if candidates:
        for p in candidates:
            thr, matches, fps, _precision = _as_point(p)
            # Prefer more matches, then fewer FPs, then lower threshold.
            key = (int(matches), -int(fps), -float(thr))
            if best_key is None or key > best_key:
                best_key = key
                best = p
        return best

    # Fallback: maximize precision, then matches, then minimize fps, then prefer lower threshold.
    best2: Optional[Dict[str, Any]] = None
    best2_key: Optional[Tuple[float, int, int, float]] = None
    for p in curve:
        if not isinstance(p, dict):
            continue
        thr, matches, fps, precision = _as_point(p)
        key = (float(precision), int(matches), -int(fps), -float(thr))
        if best2_key is None or key > best2_key:
            best2_key = key
            best2 = p
    return best2


def _select_seed_threshold_candidate_points_impl(
    curve: Sequence[Dict[str, Any]],
    *,
    max_candidates: int,
    target_precision: Optional[float] = None,
    select_operating_point_fn: Callable[..., Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Select up to max_candidates operating points from a seed-threshold curve.

    The curve is typically monotonic in (matches,fps) as threshold increases, so the Pareto frontier can
    be large. We therefore:
      - build a simple frontier in (matches, fps)
      - always include extremes + (optional) a point meeting target precision
      - downsample deterministically
    """
    try:
        max_c = max(1, int(max_candidates))
    except Exception:
        max_c = 6

    pts: List[Dict[str, Any]] = [p for p in (curve or []) if isinstance(p, dict) and p.get("threshold") is not None]
    if not pts:
        return []

    def _pt_key(p: Dict[str, Any]) -> Tuple[float, int, int, float]:
        try:
            thr = float(p.get("threshold") or 0.0)
        except Exception:
            thr = 0.0
        try:
            matches = int(p.get("matches") or 0)
        except Exception:
            matches = 0
        try:
            fps = int(p.get("fps") or 0)
        except Exception:
            fps = 0
        try:
            prec = float(p.get("precision") or 0.0)
        except Exception:
            prec = 0.0
        return (thr, matches, fps, prec)

    # Build a simple frontier in (matches,fps): keep points that improve fps as matches decreases.
    by_matches = sorted(pts, key=lambda p: (-_pt_key(p)[1], _pt_key(p)[2], -_pt_key(p)[3], _pt_key(p)[0]))
    frontier: List[Dict[str, Any]] = []
    best_fps: Optional[int] = None
    for p in by_matches:
        fps = _pt_key(p)[2]
        if best_fps is None or fps < best_fps:
            frontier.append(p)
            best_fps = fps
    frontier = sorted(frontier, key=lambda p: _pt_key(p)[0])

    chosen: List[Dict[str, Any]] = []
    chosen_thr: set[float] = set()

    def _add(p: Optional[Dict[str, Any]]) -> None:
        if not isinstance(p, dict):
            return
        thr = _pt_key(p)[0]
        thr_k = float(round(thr, 6))
        if thr_k in chosen_thr:
            return
        chosen_thr.add(thr_k)
        chosen.append(p)

    # Always include extremes (best coverage / best cleanliness).
    _add(frontier[0] if frontier else None)
    _add(frontier[-1] if frontier else None)

    # If target precision provided, include the best point meeting it (if any).
    try:
        tgt = float(target_precision) if target_precision is not None else None
    except Exception:
        tgt = None
    if tgt is not None:
        tgt = max(0.0, min(1.0, tgt))
        _add(select_operating_point_fn(frontier, min_precision=tgt))

    if len(chosen) >= max_c:
        return sorted(chosen, key=lambda p: _pt_key(p)[0])[:max_c]

    # Fill with evenly-spaced frontier points.
    n = len(frontier)
    if n > 0 and max_c > 1:
        slots = max_c
        for i in range(slots):
            if len(chosen) >= max_c:
                break
            if slots == 1:
                idx = 0
            else:
                idx = int(round(i * (n - 1) / float(slots - 1)))
            if 0 <= idx < n:
                _add(frontier[idx])

    return sorted(chosen, key=lambda p: _pt_key(p)[0])[:max_c]


def _summarize_seed_threshold_curve_for_prompt_impl(
    *,
    gt_best_scores: Mapping[Any, float],
    fp_scores: Sequence[float],
    base_seed_threshold: float,
    curve_limit: int = 48,
    build_seed_threshold_sweep_grid_fn: Callable[..., List[float]],
    compute_seed_threshold_curve_fn: Callable[..., List[Dict[str, Any]]],
    select_seed_threshold_operating_point_fn: Callable[..., Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Build a compact per-prompt seed-threshold curve and choose a default operating point.

    For now, the "recommended" point is:
      - the best matches attainable without reducing seed-stage precision below the base threshold's precision.
    """
    try:
        base_thr = float(base_seed_threshold)
    except Exception:
        base_thr = 0.05
    base_thr = max(0.0, min(1.0, base_thr))

    thresholds = build_seed_threshold_sweep_grid_fn(
        base_seed_threshold=base_thr,
        observed_scores=None,
        limit=max(8, int(curve_limit)),
    )
    curve = compute_seed_threshold_curve_fn(gt_best_scores=gt_best_scores, fp_scores=fp_scores, thresholds=thresholds)
    if curve_limit > 0 and len(curve) > int(curve_limit):
        curve = curve[: int(curve_limit)]

    base_point: Optional[Dict[str, Any]] = None
    if curve:
        base_point = min(curve, key=lambda p: abs(float(p.get("threshold") or 0.0) - base_thr))
    try:
        base_precision = float(base_point.get("precision") or 0.0) if base_point else 0.0
    except Exception:
        base_precision = 0.0

    recommended_point = select_seed_threshold_operating_point_fn(curve, min_precision=base_precision) if curve else None
    try:
        recommended_thr = float(recommended_point.get("threshold")) if recommended_point else base_thr
    except Exception:
        recommended_thr = base_thr

    return {
        "seed_threshold_base": float(base_thr),
        "seed_threshold_base_point": base_point,
        "seed_threshold_curve": curve,
        "seed_threshold_recommended": float(max(0.0, min(1.0, recommended_thr))),
        "seed_threshold_recommended_point": recommended_point,
    }


def _select_steps_from_seed_prompt_stats_impl(
    prompt_stats: Sequence[Dict[str, Any]],
    *,
    max_steps: int,
    target_precision: Optional[float] = None,
    max_candidates_per_prompt: int = 6,
    early_stop: Optional[Dict[str, Any]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    select_seed_threshold_candidate_points_fn: Callable[..., List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Greedy set-cover over GT instances, using *seed-stage* matches.

    We primarily maximize new coverage; tie-break on seed precision (higher) and fp count (lower).
    """
    try:
        max_steps_i = max(1, int(max_steps))
    except Exception:
        max_steps_i = 6

    target_precision_f: Optional[float]
    try:
        target_precision_f = float(target_precision) if target_precision is not None else None
    except Exception:
        target_precision_f = None
    if target_precision_f is not None:
        target_precision_f = max(0.0, min(1.0, target_precision_f))

    candidates: List[Dict[str, Any]] = []
    total_gt_keys: set[int] = set()
    for cand in prompt_stats:
        if not isinstance(cand, dict):
            continue
        prompt = str(cand.get("prompt") or "").strip()
        if not prompt:
            continue
        if cand.get("bg_drop"):
            if log_fn:
                try:
                    bg_rate = float(cand.get("bg_veto_rate") or 0.0)
                    bg_checked = int(cand.get("bg_checked") or 0)
                    log_fn(
                        f"[steps] prompt bg-drop: skip '{prompt}' "
                        f"(bg_veto_rate={bg_rate:.2f}, checked={bg_checked})"
                    )
                except Exception:
                    pass
            continue

        curve = cand.get("seed_threshold_curve") if isinstance(cand.get("seed_threshold_curve"), list) else None
        curve_points = curve or []
        candidate_points = select_seed_threshold_candidate_points_fn(
            curve_points,
            max_candidates=max_candidates_per_prompt,
            target_precision=target_precision_f,
        )

        # If we have no curve (or it was empty), fall back to a single implicit point.
        if not candidate_points:
            thr_fallback: Optional[float] = None
            for key in ("seed_threshold_recommended", "seed_threshold_base"):
                if cand.get(key) is None:
                    continue
                try:
                    thr_fallback = float(cand.get(key))
                    break
                except Exception:
                    continue
            if thr_fallback is None:
                thr_fallback = 0.05
            thr_fallback = max(0.0, min(1.0, float(thr_fallback)))
            candidate_points = [{"threshold": float(thr_fallback), "matches": 0, "fps": int(cand.get("fps") or 0), "precision": 0.0}]

        gt_best_scores = cand.get("gt_best_scores") if isinstance(cand.get("gt_best_scores"), dict) else {}
        for point in candidate_points:
            if not isinstance(point, dict):
                continue
            try:
                selected_thr = float(point.get("threshold"))
            except Exception:
                continue
            selected_thr = max(0.0, min(1.0, selected_thr))

            matched_keys: set[int] = set()
            if gt_best_scores:
                for k, v in gt_best_scores.items():
                    try:
                        k_int = int(k)
                    except Exception:
                        continue
                    try:
                        v_f = float(v)
                    except Exception:
                        continue
                    if v_f >= float(selected_thr):
                        matched_keys.add(k_int)
            else:
                covered = cand.get("matched_keys")
                if isinstance(covered, set):
                    matched_keys = covered

            total_gt_keys |= matched_keys

            try:
                matches = int(point.get("matches") or 0) or len(matched_keys)
            except Exception:
                matches = len(matched_keys)
            try:
                fps = int(point.get("fps") or 0)
            except Exception:
                fps = int(cand.get("fps") or 0)
            try:
                precision = float(point.get("precision") or 0.0)
            except Exception:
                precision = float(matches) / float(max(1, matches + fps))

            candidates.append(
                {
                    **cand,
                    "prompt": prompt,
                    "matched_keys": matched_keys,
                    "matches": int(matches),
                    "fps": int(fps),
                    "precision": float(precision),
                    "selected_seed_threshold": float(selected_thr),
                    "selected_seed_threshold_point": point,
                }
            )

    covered_all: set[int] = set()
    selected: List[Dict[str, Any]] = []
    selected_prompts: set[str] = set()
    remaining = candidates[:]
    early_cfg = early_stop or {"enabled": False}
    new_matches_history: List[int] = []
    total_gt = len(total_gt_keys)
    early_enabled = bool(early_cfg.get("enabled"))
    early_min_steps = int(early_cfg.get("min_steps") or 0)
    early_window = int(early_cfg.get("window") or 1)
    early_increment = float(early_cfg.get("min_increment") or 0.0)
    early_precision_floor = early_cfg.get("precision_floor")
    stop_reason: Optional[str] = None
    triggered = False
    while remaining and len(selected) < max_steps_i:
        best: Optional[Dict[str, Any]] = None
        best_key: Optional[Tuple[int, int, float, int]] = None
        for cand in remaining:
            prompt = str(cand.get("prompt") or "").strip()
            if prompt in selected_prompts:
                continue
            new = cand.get("matched_keys") - covered_all
            new_matches = len(new)
            if new_matches <= 0:
                continue
            precision = float(cand.get("precision") or 0.0)
            fps = int(cand.get("fps") or 0)
            meets = 1
            if target_precision_f is not None:
                meets = 1 if precision >= float(target_precision_f) else 0
            key = (meets, new_matches, precision, -fps)
            if best_key is None or key > best_key:
                best_key = key
                best = cand
        if best is None:
            stop_reason = "no_candidates"
            break
        if early_enabled and target_precision_f is not None and early_precision_floor is not None:
            if len(selected) >= early_min_steps and float(best.get("precision") or 0.0) < float(early_precision_floor):
                stop_reason = "precision_floor"
                triggered = True
                if log_fn:
                    try:
                        log_fn(
                            f"[steps] early-stop: best candidate precision {float(best.get('precision') or 0.0):.3f} "
                            f"below target-{float(early_cfg.get('precision_margin') or 0.0):.2f}"
                        )
                    except Exception:
                        pass
                break
        selected.append(best)
        covered_all |= best.get("matched_keys")
        selected_prompts.add(str(best.get("prompt") or "").strip())
        remaining = [c for c in remaining if str(c.get("prompt") or "").strip() not in selected_prompts]
        if best_key is not None:
            new_matches_history.append(int(best_key[1]))
        if log_fn:
            try:
                log_fn(
                    f"[steps] select step '{best.get('prompt')}' (new_gt={best_key[1] if best_key else 0}, "
                    f"text_thr={float(best.get('selected_seed_threshold') or 0.0):.3f} "
                    f"text_prec={float(best.get('precision') or 0.0):.3f}, text_fps={int(best.get('fps') or 0)})"
                )
            except Exception:
                pass
        if early_enabled and total_gt > 0 and len(selected) >= early_min_steps and early_window > 0:
            window = new_matches_history[-early_window:]
            if window:
                frac = float(sum(window)) / float(total_gt)
                if frac < float(early_increment):
                    stop_reason = "coverage_stall"
                    triggered = True
                    if log_fn:
                        try:
                            log_fn(
                                f"[steps] early-stop: coverage gain {frac * 100:.2f}% over last {len(window)} step(s) "
                                f"(threshold {float(early_increment) * 100:.2f}%)"
                            )
                        except Exception:
                            pass
                    break
    if stop_reason is None:
        if len(selected) >= max_steps_i:
            stop_reason = "max_steps"
        elif not remaining:
            stop_reason = "no_candidates"
        else:
            stop_reason = "complete"
    early_info = {
        "enabled": bool(early_enabled),
        "mode": str(early_cfg.get("mode") or "balanced"),
        "triggered": bool(triggered),
        "reason": str(stop_reason),
        "selected_steps": int(len(selected)),
        "max_steps": int(max_steps_i),
        "min_steps": int(early_min_steps),
        "window": int(early_window),
        "min_increment": float(early_increment),
        "precision_margin": float(early_cfg.get("precision_margin") or 0.0),
    }
    return selected, early_info


def _resolve_steps_early_stop_config_impl(payload: Any, *, target_precision: Optional[float]) -> Dict[str, Any]:
    enabled = bool(getattr(payload, "steps_early_stop", False))
    mode = str(getattr(payload, "steps_early_stop_mode", "balanced") or "balanced").lower().strip()
    if mode not in {"conservative", "balanced", "aggressive"}:
        mode = "balanced"
    mode_map = {
        "conservative": {"min_steps": 4, "window": 3, "min_increment": 0.002, "precision_margin": 0.15},
        "balanced": {"min_steps": 3, "window": 2, "min_increment": 0.005, "precision_margin": 0.1},
        "aggressive": {"min_steps": 2, "window": 2, "min_increment": 0.01, "precision_margin": 0.05},
    }
    cfg = mode_map[mode]
    precision_floor = None
    if target_precision is not None:
        try:
            precision_floor = max(0.0, min(1.0, float(target_precision) - float(cfg["precision_margin"])))
        except Exception:
            precision_floor = None
    return {
        "enabled": enabled,
        "mode": mode,
        "min_steps": int(cfg["min_steps"]),
        "window": int(cfg["window"]),
        "min_increment": float(cfg["min_increment"]),
        "precision_floor": precision_floor,
        "precision_margin": float(cfg["precision_margin"]),
    }


def _build_seed_stage_candidate_from_prompt_stat_impl(
    stat: Dict[str, Any],
    *,
    prompt: str,
    fallback_seed_threshold: float,
) -> Optional[Dict[str, Any]]:
    """
    Build a "selected-style" candidate dict for a single prompt from its seed-stage stats.

    This uses the prompt's recommended seed threshold (if available) as the operating point so it can be
    considered during prompt-subset refinement.
    """
    if not isinstance(stat, dict):
        return None
    prompt_s = str(prompt or "").strip()
    if not prompt_s:
        return None

    try:
        thr = float(stat.get("seed_threshold_recommended"))
    except Exception:
        try:
            thr = float(stat.get("seed_threshold_base"))
        except Exception:
            thr = float(fallback_seed_threshold)
    thr = max(0.0, min(1.0, float(thr)))

    gt_best_scores = stat.get("gt_best_scores") if isinstance(stat.get("gt_best_scores"), dict) else {}
    matched_keys: set[int] = set()
    for k, v in (gt_best_scores or {}).items():
        try:
            k_int = int(k)
        except Exception:
            continue
        try:
            v_f = float(v)
        except Exception:
            continue
        if v_f >= float(thr):
            matched_keys.add(k_int)

    curve = stat.get("seed_threshold_curve") if isinstance(stat.get("seed_threshold_curve"), list) else []
    selected_point: Optional[Dict[str, Any]] = None
    if curve:
        try:
            selected_point = min(curve, key=lambda p: abs(float(p.get("threshold") or 0.0) - float(thr)))
        except Exception:
            selected_point = None
    if selected_point is None and isinstance(stat.get("seed_threshold_recommended_point"), dict):
        selected_point = stat.get("seed_threshold_recommended_point")

    try:
        matches = int(selected_point.get("matches") or 0) if selected_point else len(matched_keys)
    except Exception:
        matches = len(matched_keys)
    try:
        fps = int(selected_point.get("fps") or 0) if selected_point else 0
    except Exception:
        fps = 0
    try:
        precision = float(selected_point.get("precision") or 0.0) if selected_point else float(matches) / float(max(1, matches + fps))
    except Exception:
        precision = float(matches) / float(max(1, matches + fps))

    return {
        **stat,
        "prompt": prompt_s,
        "matched_keys": matched_keys,
        "matches": int(matches),
        "fps": int(fps),
        "precision": float(precision),
        "selected_seed_threshold": float(thr),
        "selected_seed_threshold_point": selected_point,
    }


def _refine_steps_prompt_subset_seed_stage_impl(
    prompt_stats: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    *,
    max_steps: int,
    target_precision: Optional[float] = None,
    max_iters: int = 6,
    top_k: int = 6,
    base_seed_threshold: float = 0.05,
    log_fn: Optional[Callable[[str], None]] = None,
    build_seed_stage_candidate_fn: Callable[..., Optional[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Bounded prompt-subset refinement using only seed-stage stats (fast; no extra SAM runs).

    This is intended as a "local search" improvement on top of the greedy set-cover selection:
      - drop redundant steps (no unique GT coverage)
      - try a small number of add/swap moves guided by uncovered GT keys

    NOTE: This does not run full expansion/head tuning. It is a cheap approximation to improve step choice.
    """
    try:
        max_steps_i = max(1, int(max_steps))
    except Exception:
        max_steps_i = 6
    try:
        iters = max(0, int(max_iters))
    except Exception:
        iters = 0
    try:
        k = max(1, int(top_k))
    except Exception:
        k = 6
    try:
        base_thr = float(base_seed_threshold)
    except Exception:
        base_thr = 0.05
    base_thr = max(0.0, min(1.0, float(base_thr)))

    tgt: Optional[float]
    try:
        tgt = float(target_precision) if target_precision is not None else None
    except Exception:
        tgt = None
    if tgt is not None:
        tgt = max(0.0, min(1.0, float(tgt)))

    current: List[Dict[str, Any]] = [c for c in (selected or []) if isinstance(c, dict) and str(c.get("prompt") or "").strip()]
    if not current:
        return [], {"enabled": False, "reason": "no_selected"}

    # Build a pool of one "default operating point" per prompt for add/swap moves.
    pool: Dict[str, Dict[str, Any]] = {}
    for stat in prompt_stats or []:
        if not isinstance(stat, dict):
            continue
        p = str(stat.get("prompt") or "").strip()
        if not p:
            continue
        cand = build_seed_stage_candidate_fn(stat, prompt=p, fallback_seed_threshold=base_thr)
        if cand:
            pool[p.lower()] = cand

    start_prompts = [str(c.get("prompt") or "") for c in current]

    def _score(cands: Sequence[Dict[str, Any]]) -> Tuple[int, float, int, float, int]:
        covered: set[int] = set()
        fps_total = 0
        for c in cands:
            mk = c.get("matched_keys")
            if isinstance(mk, set):
                covered |= mk
            try:
                fps_total += int(c.get("fps") or 0)
            except Exception:
                pass
        cov = int(len(covered))
        prec = float(cov) / float(max(1, cov + int(fps_total)))
        if tgt is not None and cov > 0 and prec >= float(tgt):
            return (1, float(cov), -int(fps_total), float(prec), -len(cands))
        return (0, float(prec), int(cov), -int(fps_total), -len(cands))

    def _unique_counts(cands: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[int, int] = {}
        by_prompt: Dict[str, set[int]] = {}
        for c in cands:
            p = str(c.get("prompt") or "").strip()
            mk = c.get("matched_keys") if isinstance(c.get("matched_keys"), set) else set()
            by_prompt[p] = mk
            for k_int in mk:
                counts[int(k_int)] = counts.get(int(k_int), 0) + 1
        uniq: Dict[str, int] = {}
        for p, mk in by_prompt.items():
            uniq[p] = sum(1 for k_int in mk if counts.get(int(k_int), 0) == 1)
        return uniq

    history: List[Dict[str, Any]] = []

    # First, drop purely redundant steps deterministically (no unique coverage).
    changed = True
    while changed and len(current) > 1:
        changed = False
        uniq = _unique_counts(current)
        redundant = [c for c in current if uniq.get(str(c.get("prompt") or "").strip(), 0) <= 0]
        if not redundant:
            break
        # Drop the most "costly" redundant step first (highest fps, then lowest precision).
        redundant = sorted(
            redundant,
            key=lambda c: (
                -int(c.get("fps") or 0),
                float(c.get("precision") or 0.0),
                str(c.get("prompt") or ""),
            ),
        )
        drop = redundant[0]
        before = [str(c.get("prompt") or "") for c in current]
        current = [c for c in current if c is not drop]
        after = [str(c.get("prompt") or "") for c in current]
        history.append({"op": "drop_redundant", "dropped": str(drop.get("prompt") or ""), "before": before, "after": after})
        changed = True

    cur_key = _score(current)

    for _iter in range(iters):
        prompts_now = {str(c.get("prompt") or "").strip().lower() for c in current}
        covered_now: set[int] = set()
        for c in current:
            mk = c.get("matched_keys")
            if isinstance(mk, set):
                covered_now |= mk

        add_pool: List[Dict[str, Any]] = []
        for p_l, cand in pool.items():
            if p_l in prompts_now:
                continue
            mk = cand.get("matched_keys")
            if not isinstance(mk, set):
                continue
            new_cov = len(mk - covered_now)
            if new_cov <= 0:
                continue
            add_pool.append({**cand, "_new_cov": int(new_cov)})
        add_pool = sorted(
            add_pool,
            key=lambda c: (
                int(c.get("_new_cov") or 0),
                float(c.get("precision") or 0.0),
                -int(c.get("fps") or 0),
                str(c.get("prompt") or ""),
            ),
            reverse=True,
        )[:k]

        uniq = _unique_counts(current)
        drop_pool = sorted(
            current,
            key=lambda c: (
                uniq.get(str(c.get("prompt") or "").strip(), 0),
                -int(c.get("fps") or 0),
                float(c.get("precision") or 0.0),
                str(c.get("prompt") or ""),
            ),
        )[:k]

        best_move: Optional[Dict[str, Any]] = None
        best_next: Optional[List[Dict[str, Any]]] = None
        best_key = cur_key

        def _try(next_cands: List[Dict[str, Any]], move: Dict[str, Any]) -> None:
            nonlocal best_move, best_next, best_key
            # Enforce unique prompts + max_steps.
            if len(next_cands) > max_steps_i:
                return
            seen: set[str] = set()
            for c in next_cands:
                p = str(c.get("prompt") or "").strip().lower()
                if not p or p in seen:
                    return
                seen.add(p)
            key = _score(next_cands)
            if key > best_key:
                best_key = key
                best_next = next_cands
                best_move = move

        # Add moves (only if room).
        if len(current) < max_steps_i:
            for add in add_pool:
                _try(current + [add], {"op": "add", "added": str(add.get("prompt") or "")})

        # Swap moves.
        for add in add_pool:
            for drop in drop_pool:
                next_cands = [c for c in current if c is not drop] + [add]
                _try(
                    next_cands,
                    {"op": "swap", "added": str(add.get("prompt") or ""), "dropped": str(drop.get("prompt") or "")},
                )

        if best_next is None or best_move is None:
            break

        before = [str(c.get("prompt") or "") for c in current]
        current = best_next
        after = [str(c.get("prompt") or "") for c in current]
        history.append({**best_move, "before": before, "after": after, "key_before": cur_key, "key_after": best_key})
        cur_key = best_key

        if log_fn:
            try:
                log_fn(f"[steps] refine prompt subset: {history[-1]}")
            except Exception:
                pass

    final_prompts = [str(c.get("prompt") or "") for c in current]
    return current, {
        "enabled": True,
        "mode": "seed_stage_local_search",
        "max_iters": int(iters),
        "top_k": int(k),
        "start_steps": start_prompts,
        "final_steps": final_prompts,
        "history": history,
    }


def _build_steps_recipe_step_list_from_selected_stats_impl(
    selected_stats: Sequence[Dict[str, Any]],
    *,
    prompts_fallback: Optional[Sequence[str]],
    payload: Any,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Build the schema-v2 step list for a class, using the selected per-prompt stats.

    This is intentionally tolerant of future fields:
    - if a selected stat dict includes a per-prompt `seed_threshold` (or `selected_seed_threshold`), we
      use it for the corresponding step; otherwise we fall back to `payload.seed_threshold`.
    """
    prompts: List[str] = []
    step_list: List[Dict[str, Any]] = []

    for s in selected_stats or []:
        if not isinstance(s, dict):
            continue
        prompt = str(s.get("prompt") or "").strip()
        if prompt:
            prompts.append(prompt)

    if not prompts and prompts_fallback:
        try:
            fallback = str(list(prompts_fallback)[0]).strip()
        except Exception:
            fallback = ""
        if fallback:
            prompts = [fallback]
            selected_stats = [{}]

    def _pick_seed_threshold(stat: Dict[str, Any]) -> float:
        for key in ("seed_threshold", "selected_seed_threshold"):
            if stat.get(key) is None:
                continue
            try:
                val = float(stat.get(key))
            except Exception:
                continue
            return max(0.0, min(1.0, val))
        try:
            return max(0.0, min(1.0, float(payload.seed_threshold)))
        except Exception:
            return 0.05

    for idx, prompt in enumerate(prompts):
        stat = selected_stats[idx] if idx < len(selected_stats) and isinstance(selected_stats[idx], dict) else {}
        step_list.append(
            {
                "enabled": True,
                "prompt": prompt,
                "seed_threshold": _pick_seed_threshold(stat),
                "expand_threshold": float(payload.expand_threshold),
                "max_visual_seeds": int(getattr(payload, "steps_max_visual_seeds_per_step", 5) or 0),
                "seed_dedupe_iou": float(payload.seed_dedupe_iou),
                "dedupe_iou": float(payload.dedupe_iou),
                "max_results": int(payload.max_results),
            }
        )

    return prompts, step_list


def _resolve_steps_prompt_prefilter_config_impl(
    payload: Any,
    *,
    allow_prefilter: bool = True,
) -> Dict[str, Any]:
    requested = bool(getattr(payload, "steps_prompt_prefilter", False))
    enabled = bool(requested and allow_prefilter)
    disabled_reason = None
    if requested and not allow_prefilter:
        disabled_reason = "head_encoder_not_clip"
    mode = str(getattr(payload, "steps_prompt_prefilter_mode", "balanced") or "balanced").lower().strip()
    if mode not in {"conservative", "balanced", "aggressive"}:
        mode = "balanced"
    mode_map = {
        "conservative": {"sample_size": 20, "keep_ratio": 0.6},
        "balanced": {"sample_size": 40, "keep_ratio": 0.4},
        "aggressive": {"sample_size": 80, "keep_ratio": 0.25},
    }
    cfg = mode_map[mode]
    return {
        "enabled": enabled,
        "mode": mode,
        "sample_size": int(cfg["sample_size"]),
        "keep_ratio": float(cfg["keep_ratio"]),
        "requested": requested,
        "disabled_reason": disabled_reason,
    }


def _resolve_steps_prompt_bg_drop_config_impl(
    payload: Any,
    *,
    allow_drop: bool = True,
) -> Dict[str, Any]:
    requested = bool(getattr(payload, "steps_prompt_bg_drop", False))
    enabled = bool(requested and allow_drop)
    disabled_reason = None
    if requested and not allow_drop:
        disabled_reason = "no_background_classes"
    mode = str(getattr(payload, "steps_prompt_bg_drop_mode", "balanced") or "balanced").lower().strip()
    if mode not in {"conservative", "balanced", "aggressive"}:
        mode = "balanced"
    mode_map = {
        "conservative": {"min_checked": 60, "drop_rate": 0.75},
        "balanced": {"min_checked": 40, "drop_rate": 0.6},
        "aggressive": {"min_checked": 20, "drop_rate": 0.45},
    }
    cfg = mode_map[mode]
    return {
        "enabled": enabled,
        "mode": mode,
        "min_checked": int(cfg["min_checked"]),
        "drop_rate": float(cfg["drop_rate"]),
        "requested": requested,
        "disabled_reason": disabled_reason,
    }


def _resolve_steps_hard_negative_export_config_impl(payload: Any) -> Dict[str, Any]:
    enabled = bool(getattr(payload, "steps_hard_negative_export", False))
    try:
        max_crops = int(getattr(payload, "steps_hard_negative_max_crops", 0) or 0)
    except Exception:
        max_crops = 0
    max_crops = max(0, int(max_crops))
    try:
        min_prob = float(getattr(payload, "steps_hard_negative_min_prob", 0.0) or 0.0)
    except Exception:
        min_prob = 0.0
    min_prob = max(0.0, min(1.0, float(min_prob)))
    enabled = bool(enabled and max_crops > 0)
    return {
        "enabled": enabled,
        "max_crops": int(max_crops),
        "min_prob": float(min_prob),
    }


def _estimate_steps_speed_factor_impl(payload: Any, *, allow_prefilter: bool = True) -> float:
    early_enabled = bool(getattr(payload, "steps_early_stop", False))
    early_mode = str(getattr(payload, "steps_early_stop_mode", "balanced") or "balanced").lower().strip()
    if early_mode not in {"conservative", "balanced", "aggressive"}:
        early_mode = "balanced"
    prefilter_enabled = bool(getattr(payload, "steps_prompt_prefilter", False) and allow_prefilter)
    prefilter_mode = str(getattr(payload, "steps_prompt_prefilter_mode", "balanced") or "balanced").lower().strip()
    if prefilter_mode not in {"conservative", "balanced", "aggressive"}:
        prefilter_mode = "balanced"
    bg_drop_enabled = bool(getattr(payload, "steps_prompt_bg_drop", False))
    bg_drop_mode = str(getattr(payload, "steps_prompt_bg_drop_mode", "balanced") or "balanced").lower().strip()
    if bg_drop_mode not in {"conservative", "balanced", "aggressive"}:
        bg_drop_mode = "balanced"
    early_factor = 1.0
    if early_enabled:
        early_factor = 0.9 if early_mode == "conservative" else 0.65 if early_mode == "aggressive" else 0.8
    prefilter_factor = 1.0
    if prefilter_enabled:
        prefilter_factor = 0.85 if prefilter_mode == "conservative" else 0.55 if prefilter_mode == "aggressive" else 0.7
    bg_drop_factor = 1.0
    if bg_drop_enabled:
        bg_drop_factor = 0.92 if bg_drop_mode == "conservative" else 0.7 if bg_drop_mode == "aggressive" else 0.82
    return float(early_factor * prefilter_factor * bg_drop_factor)


def _estimate_agent_global_optimizer_image_evals_impl(
    *,
    val_images: int,
    eval_caps: Sequence[int],
    keep_ratio: float,
    rounds: int,
    max_trials: int,
    mutations_per_round: int,
) -> Tuple[int, List[int], bool]:
    parsed = []
    for cap in eval_caps or []:
        try:
            b = int(cap)
        except Exception:
            continue
        if b > 0:
            parsed.append(b)
    budgets = sorted(set(parsed))
    if not budgets:
        return 0, [], True
    try:
        val_n = max(1, int(val_images))
    except Exception:
        val_n = int(max(1, val_images))
    try:
        keep = max(0.0, min(1.0, float(keep_ratio)))
    except Exception:
        keep = 0.5
    try:
        rounds_i = max(1, int(rounds))
    except Exception:
        rounds_i = 1
    try:
        max_trials_i = max(1, int(max_trials))
    except Exception:
        max_trials_i = 1
    try:
        mutations_i = max(1, int(mutations_per_round))
    except Exception:
        mutations_i = 1
    candidates_per_round = 1
    if max_trials_i > 1:
        max_mut = max(1, min(mutations_i, max_trials_i - 1))
        candidates_per_round = 1 + max_mut
    total = 0
    for _round in range(rounds_i):
        active = max(1, candidates_per_round)
        for idx, budget in enumerate(budgets):
            eff_budget = min(int(budget), val_n)
            total += int(active) * int(eff_budget)
            if idx < len(budgets) - 1:
                active = max(1, int(math.ceil(active * keep)))
    return int(total), budgets, False


def _collect_clip_prefilter_crops_impl(
    *,
    cat_id: int,
    eval_ids: Sequence[int],
    images: Dict[int, Dict[str, Any]],
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]],
    sample_size: int,
    seed: int,
) -> List[Image.Image]:
    candidates: List[Tuple[int, List[float]]] = []
    eval_set = set(int(i) for i in eval_ids)
    for img_id, cat_map in gt_by_image_cat.items():
        if int(img_id) not in eval_set:
            continue
        bboxes = cat_map.get(int(cat_id)) or []
        for bbox in bboxes:
            if not bbox or len(bbox) < 4:
                continue
            candidates.append((int(img_id), list(bbox)))
    if not candidates:
        return []
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    sample_size = max(1, int(sample_size))
    picks = candidates[: min(sample_size, len(candidates))]
    crops: List[Image.Image] = []
    image_cache: Dict[int, Image.Image] = {}
    for img_id, bbox in picks:
        info = images.get(int(img_id)) or {}
        path = info.get("path")
        if not path:
            continue
        pil_img = image_cache.get(int(img_id))
        if pil_img is None:
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception:
                continue
            image_cache[int(img_id)] = pil_img
        try:
            x0, y0, w, h = bbox[:4]
            x1 = int(max(0.0, float(x0)))
            y1 = int(max(0.0, float(y0)))
            x2 = int(min(float(pil_img.width), float(x0) + float(w)))
            y2 = int(min(float(pil_img.height), float(y0) + float(h)))
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            crops.append(pil_img.crop((x1, y1, x2, y2)))
        except Exception:
            continue
    return crops


def _normalize_steps_for_head_tuning_impl(
    steps: Sequence[Dict[str, Any]],
    *,
    payload: Any,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        if step.get("enabled") is False:
            continue
        prompt = str(step.get("prompt") or "").strip()
        if not prompt:
            continue

        try:
            seed_thr = float(step.get("seed_threshold") if step.get("seed_threshold") is not None else payload.seed_threshold)
        except Exception:
            seed_thr = float(payload.seed_threshold)
        seed_thr = max(0.0, min(1.0, float(seed_thr)))

        try:
            expand_thr = float(
                step.get("expand_threshold") if step.get("expand_threshold") is not None else payload.expand_threshold
            )
        except Exception:
            expand_thr = float(payload.expand_threshold)
        expand_thr = max(0.0, min(1.0, float(expand_thr)))

        try:
            max_seeds = int(
                step.get("max_visual_seeds")
                if step.get("max_visual_seeds") is not None
                else getattr(payload, "steps_max_visual_seeds_per_step", 0)
            )
        except Exception:
            max_seeds = int(getattr(payload, "steps_max_visual_seeds_per_step", 0) or 0)
        max_seeds = max(0, int(max_seeds))

        try:
            seed_iou = float(step.get("seed_dedupe_iou") if step.get("seed_dedupe_iou") is not None else payload.seed_dedupe_iou)
        except Exception:
            seed_iou = float(payload.seed_dedupe_iou)
        seed_iou = max(0.0, min(1.0, float(seed_iou)))

        try:
            out_iou = float(step.get("dedupe_iou") if step.get("dedupe_iou") is not None else payload.dedupe_iou)
        except Exception:
            out_iou = float(payload.dedupe_iou)
        out_iou = max(0.0, min(1.0, float(out_iou)))

        try:
            max_results = int(step.get("max_results") if step.get("max_results") is not None else payload.max_results)
        except Exception:
            max_results = int(payload.max_results)
        max_results = max(1, int(max_results))

        out.append(
            {
                "prompt": prompt,
                "seed_threshold": float(seed_thr),
                "expand_threshold": float(expand_thr),
                "max_visual_seeds": int(max_seeds),
                "seed_dedupe_iou": float(seed_iou),
                "dedupe_iou": float(out_iou),
                "max_results": int(max_results),
            }
        )
    return out


def _prefilter_prompts_with_clip_impl(
    prompts: Sequence[str],
    *,
    keep_prompts: Sequence[str],
    cat_id: int,
    class_name: str,
    eval_ids: Sequence[int],
    images: Dict[int, Dict[str, Any]],
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]],
    clip_model_name: Optional[str],
    sample_size: int,
    keep_ratio: float,
    seed: int,
    log_fn: Optional[Callable[[str], None]] = None,
    collect_crops_fn: Callable[..., List[Image.Image]],
    encode_pil_fn: Callable[..., Optional[np.ndarray]],
    encode_text_fn: Callable[..., Optional[np.ndarray]],
) -> List[str]:
    prompt_list: List[str] = []
    seen: set[str] = set()
    for p in prompts:
        key = str(p).strip()
        if not key:
            continue
        low = key.lower()
        if low in seen:
            continue
        seen.add(low)
        prompt_list.append(key)
    if not prompt_list:
        return []

    keep_set = {str(p).strip().lower() for p in keep_prompts if str(p).strip()}
    extra_indices = [idx for idx, p in enumerate(prompt_list) if p.lower() not in keep_set]
    if not extra_indices:
        return prompt_list

    crops = collect_crops_fn(
        cat_id=cat_id,
        eval_ids=eval_ids,
        images=images,
        gt_by_image_cat=gt_by_image_cat,
        sample_size=sample_size,
        seed=seed,
    )
    if not crops:
        if log_fn:
            try:
                log_fn(f"[steps] CLIP prefilter skipped for {class_name}: no GT crops in sample.")
            except Exception:
                pass
        return prompt_list

    img_emb = encode_pil_fn(crops, clip_model_override=clip_model_name)
    text_emb = encode_text_fn(prompt_list, clip_model_override=clip_model_name)
    if img_emb is None or text_emb is None:
        if log_fn:
            try:
                log_fn(f"[steps] CLIP prefilter skipped for {class_name}: CLIP embeddings unavailable.")
            except Exception:
                pass
        return prompt_list

    try:
        sims = np.matmul(text_emb, img_emb.T)
        scores = sims.max(axis=1) if sims.size else np.zeros(len(prompt_list), dtype=np.float32)
    except Exception:
        if log_fn:
            try:
                log_fn(f"[steps] CLIP prefilter skipped for {class_name}: similarity computation failed.")
            except Exception:
                pass
        return prompt_list

    keep_ratio = max(0.05, min(float(keep_ratio), 1.0))
    extra_count = len(extra_indices)
    keep_count = max(1, int(round(extra_count * keep_ratio)))
    keep_count = min(extra_count, keep_count)
    ranked = sorted(extra_indices, key=lambda i: float(scores[i]), reverse=True)
    kept_extra = set(ranked[:keep_count])

    filtered = [p for idx, p in enumerate(prompt_list) if idx in kept_extra or p.lower() in keep_set]
    if log_fn:
        try:
            log_fn(
                f"[steps] CLIP prefilter {class_name}: kept {len(filtered)}/{len(prompt_list)} prompts "
                f"(base {len(prompt_list) - extra_count} + filtered {len(kept_extra)})"
            )
        except Exception:
            pass
    return filtered


def _export_hard_negative_replay_impl(
    *,
    dataset_id: str,
    class_id: int,
    class_name: str,
    entries: Sequence[Dict[str, Any]],
    max_crops: int,
    replay_root: Path,
    path_is_within_root_fn: Callable[[Path, Path], bool],
    time_fn: Callable[[], float],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    if max_crops <= 0 or not entries:
        return None
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(class_name or "").strip()).strip("_")
    if not safe_name:
        safe_name = f"class_{int(class_id)}"
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = (replay_root / str(dataset_id) / f"{int(class_id):03d}_{safe_name}" / stamp).resolve()
    if not path_is_within_root_fn(run_dir, replay_root):
        return None
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    def _score(entry: Dict[str, Any]) -> float:
        try:
            return float(entry.get("score") or 0.0)
        except Exception:
            return 0.0

    entries_sorted = sorted([e for e in entries if isinstance(e, dict)], key=_score, reverse=True)
    seen_keys: set[Tuple[int, Tuple[float, float, float, float]]] = set()
    saved: List[Dict[str, Any]] = []
    for entry in entries_sorted:
        if len(saved) >= int(max_crops):
            break
        try:
            img_id = int(entry.get("image_id"))
        except Exception:
            continue
        bbox = entry.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            bbox_key = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:
            continue
        dedupe_key = (int(img_id), bbox_key)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        image_path = entry.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            continue
        try:
            with Image.open(image_path) as img:
                pil_img = img.convert("RGB")
        except Exception:
            continue
        x1, y1, x2, y2 = bbox_key
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            crop = pil_img.crop((x1, y1, x2, y2))
        except Exception:
            continue
        filename = f"hn_{len(saved):05d}.png"
        crop_path = crops_dir / filename
        try:
            crop.save(crop_path, format="PNG")
        except Exception:
            continue
        saved.append(
            {
                "image_id": int(img_id),
                "image_path": str(image_path),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(entry.get("score") or 0.0),
                "clip_prob": entry.get("clip_prob"),
                "clip_bg_prob": entry.get("clip_bg_prob"),
                "clip_margin": entry.get("clip_margin"),
                "prompt": entry.get("prompt"),
                "crop_path": str(Path("crops") / crop_path.name),
            }
        )

    if not saved:
        return None

    manifest = {
        "dataset_id": str(dataset_id),
        "class_id": int(class_id),
        "class_name": str(class_name or f"class_{class_id}"),
        "created_at": float(time_fn()),
        "count": int(len(saved)),
        "entries": saved,
    }
    try:
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)
    except Exception:
        return None
    if log_fn:
        try:
            log_fn(f"[steps] Hard-negative export: saved {len(saved)}/{len(entries_sorted)} crops to {run_dir}")
        except Exception:
            pass
    return {
        "enabled": True,
        "count": int(len(saved)),
        "max_crops": int(max_crops),
        "root": str(run_dir),
        "manifest": str(run_dir / "manifest.json"),
    }
