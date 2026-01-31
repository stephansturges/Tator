from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Literal, List

from fastapi import HTTPException
from starlette.status import HTTP_404_NOT_FOUND
import math
from PIL import Image


def _write_prepass_recipe_meta(recipe_dir: Path, payload: Dict[str, Any]) -> None:
    meta_path = recipe_dir / "prepass.meta.json"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_prepass_recipe_meta(recipe_dir: Path) -> Dict[str, Any]:
    meta_path = recipe_dir / "prepass.meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prepass_recipe_not_found")
    return json.loads(meta_path.read_text())


def _parse_agent_recipe_schema_version(recipe_obj: Dict[str, Any]) -> Optional[int]:
    raw = recipe_obj.get("schema_version")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _classify_agent_recipe_mode(recipe_obj: Dict[str, Any]) -> Literal["sam3_steps", "sam3_greedy", "legacy_steps"]:
    """
    Classify an agent recipe into one of:
    - sam3_steps: explicit multi-step recipe (schema_version=2 or mode=sam3_steps)
    - sam3_greedy: prompt-bank + crop-bank / head recipe (legacy "greedy" semantics)
    - legacy_steps: older prompt-recipe style "steps" recipes (threshold per prompt/exemplar)
    """
    schema_version = _parse_agent_recipe_schema_version(recipe_obj)
    mode_raw = recipe_obj.get("mode")
    mode = mode_raw.strip() if isinstance(mode_raw, str) else None
    if mode == "sam3_steps" or schema_version == 2:
        return "sam3_steps"
    if mode == "sam3_greedy":
        return "sam3_greedy"
    # Back-compat: older greedy recipes may omit mode, but include prompt/crop bank fields.
    if isinstance(recipe_obj.get("text_prompts"), list) or isinstance(recipe_obj.get("positives"), list):
        return "sam3_greedy"
    return "legacy_steps"


def _normalize_agent_recipe_execution_plan(recipe_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an agent recipe into a lightweight execution plan.

    This is intentionally shallow and does not change behavior by itself; it exists so future
    inference/mining code paths can share a single normalization step.
    """
    mode = _classify_agent_recipe_mode(recipe_obj)
    schema_version = _parse_agent_recipe_schema_version(recipe_obj)
    return {"mode": mode, "schema_version": schema_version, "recipe": recipe_obj}


def _validate_agent_recipe_structure(recipe_obj: Dict[str, Any]) -> None:
    """Lightweight schema guard to avoid accepting malformed recipes."""
    if not isinstance(recipe_obj, dict):
        raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
    mode = _classify_agent_recipe_mode(recipe_obj)

    if mode == "sam3_steps":
        # Schema v2 (step-based) recipe: explicit steps. May optionally embed crop banks / clip head.
        steps = recipe_obj.get("steps")
        if not isinstance(steps, list) or not steps:
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        for step in steps:
            if not isinstance(step, dict):
                raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
            if "enabled" in step and not isinstance(step.get("enabled"), bool):
                raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
            prompt = step.get("prompt")
            prompts = step.get("prompts")
            if prompt is not None and not isinstance(prompt, str):
                raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
            if prompts is not None:
                if not isinstance(prompts, list) or any(not isinstance(p, str) for p in prompts):
                    raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
            has_any_prompt = bool(
                (isinstance(prompt, str) and prompt.strip())
                or (isinstance(prompts, list) and any(str(p).strip() for p in prompts))
            )
            if not has_any_prompt:
                raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")

            # Optional numeric fields (kept permissive; more detailed validation happens in execution code).
            for key in (
                "seed_threshold",
                "expand_threshold",
                "mask_threshold",
                "seed_dedupe_iou",
                "step_dedupe_iou",
                "dedupe_iou",
            ):
                if key not in step or step.get(key) is None:
                    continue
                try:
                    v = float(step.get(key))
                except Exception:
                    raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
                if math.isnan(v) or v < 0.0 or v > 1.0:
                    raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
            for key in ("max_visual_seeds", "expand_max_results", "max_results"):
                if key not in step or step.get(key) is None:
                    continue
                try:
                    iv = int(step.get(key))
                except Exception:
                    raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
                if iv < 0:
                    raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")

            for clip_key in ("clip_seed", "clip_final"):
                if clip_key not in step or step.get(clip_key) is None:
                    continue
                if not isinstance(step.get(clip_key), dict):
                    raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        positives = recipe_obj.get("positives")
        negatives = recipe_obj.get("negatives")
        if positives is not None and not isinstance(positives, list):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        if negatives is not None and not isinstance(negatives, list):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        clip_head = recipe_obj.get("clip_head")
        if clip_head is not None and not isinstance(clip_head, dict):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        params = recipe_obj.get("params")
        if params is not None and not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        return

    # New greedy recipe format: prompt bank + positive/negative crop banks.
    if mode == "sam3_greedy":
        text_prompts = recipe_obj.get("text_prompts")
        positives = recipe_obj.get("positives")
        negatives = recipe_obj.get("negatives")
        steps = recipe_obj.get("steps")
        if text_prompts is not None and not isinstance(text_prompts, list):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        if positives is not None and not isinstance(positives, list):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        if negatives is not None and not isinstance(negatives, list):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        if steps is not None and not isinstance(steps, list):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")
        return

    # Legacy prompt-list recipe format.
    if mode == "legacy_steps":
        steps = recipe_obj.get("steps")
        if steps is not None and not isinstance(steps, list):
            raise HTTPException(status_code=400, detail="agent_recipe_invalid_schema")


def _save_exemplar_crop_impl(
    *,
    exemplar: Dict[str, Any],
    images: Dict[int, Dict[str, Any]],
    crop_dir: Path,
    step_idx: int,
    crop_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Persist a single exemplar crop to disk and return enriched metadata."""
    img_id = exemplar.get("image_id")
    if img_id is None:
        return None
    info = images.get(int(img_id))
    if not info:
        return None
    bbox = exemplar.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    try:
        x, y, w, h = map(float, bbox[:4])
    except Exception:
        return None
    try:
        img_path = info.get("path")
        if not img_path:
            return None
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            width, height = pil_img.width, pil_img.height
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + w)
            y1 = min(height, y + h)
            crop = pil_img.crop((x0, y0, x1, y1))
            crop_dir.mkdir(parents=True, exist_ok=True)
            filename = crop_name or f"step_{step_idx:02d}_exemplar.png"
            crop_path = crop_dir / filename
            crop.save(crop_path, format="PNG")
    except Exception:
        return None
    bbox_norm = None
    try:
        bbox_norm = [x / width, y / height, w / width, h / height]
    except Exception:
        bbox_norm = None
    enriched = {
        **exemplar,
        "bbox": [x, y, w, h],
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_norm": bbox_norm,
        "image_size": [width, height],
        "crop_path": str(Path("crops") / crop_path.name),
        "crop_size": [crop.width, crop.height],
    }
    return enriched
