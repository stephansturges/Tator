"""Calibration metrics helpers (prompt evaluation + IoU scoring)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from PIL import Image


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











































