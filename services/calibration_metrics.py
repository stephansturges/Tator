from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
