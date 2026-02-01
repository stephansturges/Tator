"""Prepass recipe helpers."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import time
import uuid
import zipfile
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Literal, List, Tuple

from fastapi import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_412_PRECONDITION_FAILED,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
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


def _normalize_agent_recipe_steps_impl(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    extra_keys = {
        "similarity_score",
        "seed_prompt",
        "fps",
        "gain",
        "source",
        "precision",
        "recall",
        "coverage",
        "duplicates",
    }
    for step in steps:
        prompt = step.get("prompt")
        threshold = step.get("threshold")
        has_exemplar = step.get("exemplar") is not None
        if (prompt is None and not has_exemplar) or threshold is None:
            continue
        try:
            thr_val = float(threshold)
        except Exception:
            continue
        if math.isnan(thr_val) or thr_val < 0.0 or thr_val > 1.0:
            continue
        entry = {
            "prompt": "" if prompt is None else str(prompt),
            "threshold": thr_val,
            "type": step.get("type"),
            "exemplar": dict(step["exemplar"]) if isinstance(step.get("exemplar"), dict) else step.get("exemplar"),
        }
        sim_raw = step.get("similarity_score")
        if sim_raw is not None:
            try:
                sim_val = float(sim_raw)
            except Exception:
                sim_val = None
            if sim_val is not None and 0.0 <= sim_val <= 1.0:
                entry["similarity_score"] = sim_val
        for key in extra_keys:
            if key in entry:
                continue
            if key in step:
                entry[key] = step[key]
        normalized.append(entry)
    return normalized


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


def _delete_agent_recipe_impl(
    recipe_id: str,
    *,
    recipes_root: Path,
    path_is_within_root_fn,
    http_exception_cls,
) -> None:
    json_path = (recipes_root / f"{recipe_id}.json").resolve()
    zip_path = (recipes_root / f"{recipe_id}.zip").resolve()
    recipe_dir = (recipes_root / recipe_id).resolve()
    if not path_is_within_root_fn(json_path, recipes_root.resolve()):
        raise http_exception_cls(status_code=400, detail="agent_recipe_path_invalid")
    removed_any = False
    for path in (json_path, zip_path):
        if path.exists():
            try:
                path.unlink()
                removed_any = True
            except Exception:
                pass
    if recipe_dir.exists() and recipe_dir.is_dir():
        try:
            shutil.rmtree(recipe_dir)
            removed_any = True
        except Exception:
            pass
    if not removed_any:
        raise http_exception_cls(status_code=404, detail="agent_recipe_not_found")


def _list_agent_recipes_impl(
    *,
    recipes_root: Path,
    dataset_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    recipes: List[Dict[str, Any]] = []
    for path in recipes_root.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if dataset_id and data.get("dataset_id") != dataset_id:
                continue
            data["_path"] = str(path)
            zip_path = (recipes_root / f"{data.get('id','')}.zip").resolve()
            if zip_path.exists():
                data["_zip"] = str(zip_path)
            recipes.append(data)
        except Exception:
            continue
    recipes.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return recipes


def _persist_agent_recipe_impl(
    dataset_id: Optional[str],
    class_id: Optional[int],
    class_name: Optional[str],
    label: str,
    recipe: Dict[str, Any],
    *,
    crop_overrides: Optional[Dict[str, bytes]] = None,
    clip_head_overrides: Optional[Dict[str, bytes]] = None,
    meta_overrides: Optional[Dict[str, Any]] = None,
    recipes_root: Path,
    max_clip_head_bytes: int,
    max_crops: int,
    max_crop_bytes: int,
    resolve_dataset_fn,
    load_coco_index_fn,
    compute_dataset_signature_fn,
    compute_labelmap_hash_fn,
    resolve_clip_classifier_fn,
    load_clip_head_fn,
    save_clip_head_artifacts_fn,
    load_clip_head_artifacts_fn,
    save_exemplar_crop_fn,
    sanitize_prompts_fn,
    path_is_within_root_fn,
) -> Dict[str, Any]:
    if not isinstance(recipe, dict) or not recipe:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_empty")
    # Accept either a raw recipe body, or a wrapper containing {"recipe": {...}} (e.g., imported payload).
    recipe_body: Dict[str, Any] = recipe
    if (
        isinstance(recipe.get("recipe"), dict)
        and not any(k in recipe for k in ("steps", "text_prompts", "positives", "mode"))
    ):
        recipe_body = recipe.get("recipe") or {}
    if not isinstance(recipe_body, dict) or not recipe_body:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_empty")
    _validate_agent_recipe_structure(recipe_body)
    cleaned_label = label.strip() or "agent_recipe"
    recipe_id = f"ar_{uuid.uuid4().hex[:8]}"
    images: Dict[int, Dict[str, Any]] = {}
    categories: List[Dict[str, Any]] = []
    dataset_signature: Optional[str] = None
    labelmap_hash: Optional[str] = None
    labelmap_entries: Optional[List[str]] = None
    dataset_root: Optional[Path] = None
    dataset_id_clean = (dataset_id or "").strip()
    try:
        if dataset_id_clean:
            dataset_root = resolve_dataset_fn(dataset_id_clean)
            coco, _, images = load_coco_index_fn(dataset_root)
            categories = coco.get("categories") or []
            dataset_signature = compute_dataset_signature_fn(dataset_id_clean, dataset_root, images, categories)
            labelmap_hash, labelmap_entries = compute_labelmap_hash_fn(categories)
            if class_id is not None:
                try:
                    cid = int(class_id)
                except Exception:
                    cid = None
                if cid is not None:
                    found = any(int(cat.get("id", idx)) == cid for idx, cat in enumerate(categories))
                    if not found and not crop_overrides:
                        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="agent_recipe_class_missing")
    except HTTPException:
        if not crop_overrides and not meta_overrides:
            raise
    except Exception:
        # Allow portability when importing with embedded crops; we'll fall back to meta overrides.
        pass
    if not dataset_signature and meta_overrides:
        dataset_signature = meta_overrides.get("dataset_signature")
    if not labelmap_hash and meta_overrides:
        labelmap_hash = meta_overrides.get("labelmap_hash")
        labelmap_entries = meta_overrides.get("labelmap")
    if not labelmap_entries:
        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="agent_recipe_labelmap_missing")
    recipe_mode = _classify_agent_recipe_mode(recipe_body)
    if recipe_mode == "sam3_steps":
        raw_steps = recipe_body.get("steps")
        if raw_steps is None:
            raw_steps = []
        if not isinstance(raw_steps, list):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        steps_raw = [dict(s) for s in raw_steps if isinstance(s, dict)]
    else:
        steps_raw = _normalize_agent_recipe_steps_impl(recipe_body.get("steps") or [])
    text_prompts_raw = recipe_body.get("text_prompts")
    positives_raw = recipe_body.get("positives")
    negatives_raw = recipe_body.get("negatives")
    if text_prompts_raw is None:
        text_prompts_raw = recipe.get("text_prompts")
    if positives_raw is None:
        positives_raw = recipe.get("positives")
    if negatives_raw is None:
        negatives_raw = recipe.get("negatives")
    text_prompts: List[str] = []
    if isinstance(text_prompts_raw, list):
        text_prompts = sanitize_prompts_fn([str(p) for p in text_prompts_raw if str(p).strip()])
    positives_list: List[Dict[str, Any]] = (
        [p for p in (positives_raw or []) if isinstance(p, dict)] if isinstance(positives_raw, list) else []
    )
    negatives_list: List[Dict[str, Any]] = (
        [n for n in (negatives_raw or []) if isinstance(n, dict)] if isinstance(negatives_raw, list) else []
    )
    is_greedy = bool(recipe_body.get("mode") == "sam3_greedy" or text_prompts or positives_list)
    if not (steps_raw or text_prompts or positives_list):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_empty")
    recipe_dir = recipes_root / recipe_id
    cleanup_recipe_dir = True
    try:
        crops_dir = recipe_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        # Optional portable CLIP head artifacts (embedded into the recipe package).
        clip_head_cfg_raw: Optional[Dict[str, Any]] = None
        if isinstance(recipe_body.get("clip_head"), dict):
            clip_head_cfg_raw = recipe_body.get("clip_head")
        elif isinstance(recipe.get("clip_head"), dict):
            clip_head_cfg_raw = recipe.get("clip_head")

        clip_head_classifier_path: Optional[str] = None
        for src in (recipe_body, recipe):
            if isinstance(src, dict) and isinstance(src.get("_clip_head_classifier_path"), str):
                clip_head_classifier_path = str(src.get("_clip_head_classifier_path"))
                break

        clip_head_written = False
        clip_dir = recipe_dir / "clip_head"
        head_npz_bytes = None
        head_meta_bytes = None
        if clip_head_overrides:
            head_npz_bytes = clip_head_overrides.get("clip_head/head.npz") or clip_head_overrides.get("head.npz")
            head_meta_bytes = clip_head_overrides.get("clip_head/meta.json") or clip_head_overrides.get("meta.json")
        if head_npz_bytes:
            try:
                clip_dir.mkdir(parents=True, exist_ok=True)
                (clip_dir / "head.npz").write_bytes(head_npz_bytes)
                clip_head_written = True
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"agent_recipe_clip_head_write_failed:{exc}",
                ) from exc
        if head_meta_bytes:
            try:
                clip_dir.mkdir(parents=True, exist_ok=True)
                (clip_dir / "meta.json").write_bytes(head_meta_bytes)
                clip_head_written = True
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"agent_recipe_clip_head_meta_write_failed:{exc}",
                ) from exc
        if not clip_head_written and clip_head_classifier_path:
            resolved_classifier = resolve_clip_classifier_fn(clip_head_classifier_path)
            if resolved_classifier is not None:
                head = load_clip_head_fn(resolved_classifier)
                if head is not None:
                    min_prob = 0.5
                    margin = 0.0
                    if clip_head_cfg_raw:
                        try:
                            if clip_head_cfg_raw.get("min_prob") is not None:
                                min_prob = float(clip_head_cfg_raw.get("min_prob"))
                            if clip_head_cfg_raw.get("margin") is not None:
                                margin = float(clip_head_cfg_raw.get("margin"))
                        except Exception:
                            min_prob = 0.5
                            margin = 0.0
                    save_clip_head_artifacts_fn(recipe_dir=recipe_dir, head=head, min_prob=min_prob, margin=margin)
                    clip_head_written = True

        clip_head_cfg_clean: Optional[Dict[str, Any]] = None
        if clip_head_written:
            loaded = load_clip_head_artifacts_fn(recipe_dir=recipe_dir, fallback_meta=clip_head_cfg_raw)
            if loaded is not None:
                min_prob = 0.5
                margin = 0.0
                if loaded.get("min_prob") is not None:
                    try:
                        min_prob = float(loaded.get("min_prob"))
                    except Exception:
                        min_prob = 0.5
                if loaded.get("margin") is not None:
                    try:
                        margin = float(loaded.get("margin"))
                    except Exception:
                        margin = 0.0
                clip_head_cfg_clean = {
                    "artifact": "clip_head/head.npz",
                    "clip_model": loaded.get("clip_model"),
                    "proba_mode": loaded.get("proba_mode"),
                    "classes": loaded.get("classes") if isinstance(loaded.get("classes"), list) else [],
                    "min_prob": float(max(0.0, min(1.0, min_prob))),
                    "margin": float(max(0.0, min(1.0, margin))),
                }
                if clip_head_cfg_raw:
                    if clip_head_cfg_raw.get("auto_tuned") is not None:
                        clip_head_cfg_clean["auto_tuned"] = bool(clip_head_cfg_raw.get("auto_tuned"))
                    if clip_head_cfg_raw.get("target_precision") is not None:
                        try:
                            clip_head_cfg_clean["target_precision"] = float(clip_head_cfg_raw.get("target_precision"))
                        except Exception:
                            pass
            try:
                total = 0
                if clip_dir.exists():
                    for f in clip_dir.iterdir():
                        if f.is_file():
                            total += f.stat().st_size
                if total > max_clip_head_bytes:
                    raise HTTPException(
                        status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="agent_recipe_clip_head_too_large",
                    )
            except HTTPException:
                raise
            except Exception:
                pass

        def _safe_crop_filename(preferred: Optional[str], prefix: str, idx: int) -> str:
            try:
                name = Path(str(preferred)).name if preferred else ""
            except Exception:
                name = ""
            if not name:
                name = f"{prefix}_{idx:03d}.png"
            if not name.lower().endswith(".png"):
                name = f"{name}.png"
            name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            base, ext = os.path.splitext(name)
            if not ext:
                ext = ".png"
            candidate = f"{base}{ext}"
            counter = 1
            while (crops_dir / candidate).exists():
                counter += 1
                candidate = f"{base}_{counter}{ext}"
            return candidate

        def _materialize_crop_entry(
            entry: Dict[str, Any], *, prefix: str, idx: int, fallback_step_idx: int
        ) -> Optional[Dict[str, Any]]:
            """Materialize a crop into crops_dir and return a portable entry dict."""
            entry_copy = dict(entry)
            entry_copy.pop("crop_base64", None)
            crop_key = entry_copy.get("crop_path") or entry_copy.get("path")
            crop_bytes = None
            if crop_overrides and crop_key:
                crop_bytes = crop_overrides.get(str(crop_key))
                if crop_bytes is None:
                    try:
                        crop_bytes = crop_overrides.get(str(Path("crops") / Path(str(crop_key)).name))
                    except Exception:
                        crop_bytes = None
            filename = _safe_crop_filename(str(crop_key) if crop_key else None, prefix, idx)
            if crop_bytes is not None:
                crop_path = crops_dir / filename
                try:
                    with crop_path.open("wb") as fp:
                        fp.write(crop_bytes)
                    entry_copy["crop_path"] = str(Path("crops") / crop_path.name)
                    entry_copy.pop("path", None)
                    entry_copy.pop("embed_id", None)
                    entry_copy.pop("crop_base64", None)
                    return entry_copy
                except Exception:
                    # Fall back to a portable dict without guarantees if write fails.
                    entry_copy.pop("path", None)
                    entry_copy.pop("embed_id", None)
                    entry_copy.pop("crop_base64", None)
                    return entry_copy
            enriched = None
            if images:
                enriched = save_exemplar_crop_fn(
                    exemplar=entry_copy,
                    images=images,
                    crop_dir=crops_dir,
                    step_idx=fallback_step_idx,
                    crop_name=filename,
                )
            if enriched is None:
                entry_copy.pop("path", None)
                entry_copy.pop("embed_id", None)
                entry_copy.pop("crop_base64", None)
                # Ensure crop_path, if present, is made portable.
                if crop_key:
                    try:
                        entry_copy["crop_path"] = str(Path("crops") / Path(str(crop_key)).name)
                    except Exception:
                        pass
                return entry_copy
            enriched.pop("path", None)
            enriched.pop("embed_id", None)
            enriched.pop("crop_base64", None)
            return enriched

        portable_steps: List[Dict[str, Any]] = []
        portable_positives: List[Dict[str, Any]] = []
        portable_negatives: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps_raw, start=1):
            entry = dict(step)
            ex = step.get("exemplar")
            if ex:
                enriched = None
                # Prefer provided crops if present (e.g., imported package), else derive from dataset.
                crop_key = None
                if isinstance(ex, dict):
                    crop_key = ex.get("crop_path")
                    crop_bytes = None
                    if crop_overrides and crop_key:
                        crop_bytes = crop_overrides.get(crop_key)
                        if crop_bytes is None:
                            try:
                                alt_key = str(Path("crops") / Path(crop_key).name)
                                crop_bytes = crop_overrides.get(alt_key)
                            except Exception:
                                crop_bytes = None
                    if crop_bytes is not None:
                        crop_path = crops_dir / Path(crop_key).name
                        try:
                            with crop_path.open("wb") as fp:
                                fp.write(crop_bytes)
                            if crop_path.exists():
                                pass
                            enriched = {
                                **ex,
                                "crop_path": str(Path("crops") / crop_path.name),
                            }
                        except Exception:
                            enriched = dict(ex)
                if enriched is None and images and isinstance(ex, dict):
                    enriched = save_exemplar_crop_fn(exemplar=ex, images=images, crop_dir=crops_dir, step_idx=idx)
                if enriched is None and isinstance(ex, dict):
                    enriched = dict(ex)
                entry["exemplar"] = enriched
            portable_steps.append(entry)

        # Greedy-mode crop banks.
        if is_greedy and positives_list:
            for p_idx, pos in enumerate(positives_list, start=1):
                enriched_pos = _materialize_crop_entry(pos, prefix="pos", idx=p_idx, fallback_step_idx=2000 + p_idx)
                if enriched_pos:
                    portable_positives.append(enriched_pos)
        for n_idx, neg in enumerate(negatives_list, start=1):
            enriched_neg = _materialize_crop_entry(neg, prefix="neg", idx=n_idx, fallback_step_idx=3000 + n_idx)
            if enriched_neg:
                portable_negatives.append(enriched_neg)

        def _normalize_crop_path(path_str: Optional[str]) -> Optional[str]:
            if not path_str:
                return None
            try:
                return str(Path("crops") / Path(path_str).name)
            except Exception:
                return None

        # Normalize crop paths in steps and negatives for portability.
        for entry in portable_steps:
            ex = entry.get("exemplar")
            if isinstance(ex, dict) and ex.get("crop_path"):
                normalized = _normalize_crop_path(ex.get("crop_path"))
                if normalized:
                    ex["crop_path"] = normalized
                ex.pop("path", None)
                ex.pop("crop_base64", None)
        for entry in portable_positives:
            if isinstance(entry, dict) and entry.get("crop_path"):
                normalized = _normalize_crop_path(entry.get("crop_path"))
                if normalized:
                    entry["crop_path"] = normalized
                entry.pop("path", None)
                entry.pop("crop_base64", None)
        for neg in portable_negatives:
            if isinstance(neg, dict) and neg.get("crop_path"):
                normalized = _normalize_crop_path(neg.get("crop_path"))
                if normalized:
                    neg["crop_path"] = normalized
                neg.pop("path", None)
                neg.pop("crop_base64", None)

        # Enforce crop count/byte limits after all crops are materialized.
        def _assert_crop_limits() -> None:
            if not crops_dir.exists():
                return
            count = 0
            total = 0
            try:
                for cf in crops_dir.glob("*.png"):
                    count += 1
                    try:
                        total += cf.stat().st_size
                    except Exception:
                        continue
                if count > max_crops or total > max_crop_bytes:
                    raise HTTPException(
                        status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="agent_recipe_crops_too_large",
                    )
            except HTTPException as exc:
                raise exc

        _assert_crop_limits()

        params_src = recipe_body.get("params")
        if not isinstance(params_src, dict):
            params_src = recipe.get("params") if isinstance(recipe.get("params"), dict) else None
        params = params_src or {
            "mask_threshold": recipe_body.get("mask_threshold", recipe.get("mask_threshold")),
            "min_size": recipe_body.get("min_size", recipe.get("min_size")),
            "simplify_epsilon": recipe_body.get("simplify_epsilon", recipe.get("simplify_epsilon")),
            "max_results": recipe_body.get("max_results", recipe.get("max_results")),
            "similarity_score": recipe_body.get("similarity_score", recipe.get("similarity_score")),
            "seed_threshold": recipe_body.get("seed_threshold", recipe.get("seed_threshold")),
            "expand_threshold": recipe_body.get("expand_threshold", recipe.get("expand_threshold")),
            "max_visual_seeds": recipe_body.get("max_visual_seeds", recipe.get("max_visual_seeds")),
            "seed_dedupe_iou": recipe_body.get("seed_dedupe_iou", recipe.get("seed_dedupe_iou")),
            "dedupe_iou": recipe_body.get("dedupe_iou", recipe.get("dedupe_iou")),
            "use_clip_fp_guard": recipe_body.get("use_clip_fp_guard", recipe.get("use_clip_fp_guard")),
            "use_negative_exemplars": recipe_body.get("use_negative_exemplars", recipe.get("use_negative_exemplars")),
            "negative_strength": recipe_body.get("negative_strength", recipe.get("negative_strength")),
            "clip_head_background_guard": recipe_body.get(
                "clip_head_background_guard", recipe.get("clip_head_background_guard")
            ),
            "clip_head_background_margin": recipe_body.get(
                "clip_head_background_margin", recipe.get("clip_head_background_margin")
            ),
            "clip_head_background_apply": recipe_body.get(
                "clip_head_background_apply", recipe.get("clip_head_background_apply")
            ),
            "clip_head_background_penalty": recipe_body.get(
                "clip_head_background_penalty", recipe.get("clip_head_background_penalty")
            ),
        }
        if isinstance(params, dict) and "clip_head_background_guard" not in params:
            params["clip_head_background_guard"] = recipe_body.get(
                "clip_head_background_guard", recipe.get("clip_head_background_guard")
            )
        if isinstance(params, dict) and "clip_head_background_margin" not in params:
            params["clip_head_background_margin"] = recipe_body.get(
                "clip_head_background_margin", recipe.get("clip_head_background_margin")
            )
        if isinstance(params, dict) and "clip_head_background_apply" not in params:
            params["clip_head_background_apply"] = recipe_body.get(
                "clip_head_background_apply", recipe.get("clip_head_background_apply")
            )
        if isinstance(params, dict) and "clip_head_background_penalty" not in params:
            params["clip_head_background_penalty"] = recipe_body.get(
                "clip_head_background_penalty", recipe.get("clip_head_background_penalty")
            )
        thresholds = sorted({float(s.get("threshold", 0.0)) for s in portable_steps if s.get("threshold") is not None})
        if thresholds:
            params["thresholds"] = thresholds
        schema_version_out: Optional[int] = None
        try:
            schema_version_out = int(recipe_body.get("schema_version")) if recipe_body.get("schema_version") is not None else None
        except Exception:
            schema_version_out = None
        mode_out: Optional[str] = recipe_body.get("mode") if isinstance(recipe_body.get("mode"), str) else None
        if recipe_mode == "sam3_steps":
            schema_version_out = 2
            mode_out = "sam3_steps"
        elif is_greedy:
            mode_out = mode_out or "sam3_greedy"
        optimizer_raw: Optional[Dict[str, Any]] = None
        for src in (recipe_body, recipe):
            if isinstance(src, dict) and isinstance(src.get("optimizer"), dict):
                optimizer_raw = src.get("optimizer")  # type: ignore[assignment]
                break
        optimizer_clean: Optional[Dict[str, Any]] = dict(optimizer_raw) if isinstance(optimizer_raw, dict) else None
        payload = {
            "id": recipe_id,
            "dataset_id": dataset_id,
            "dataset_signature": dataset_signature,
            "labelmap_hash": labelmap_hash,
            "labelmap": labelmap_entries,
            "class_id": class_id,
            "class_name": class_name,
            "label": cleaned_label,
            "created_at": time.time(),
            "params": params,
            "recipe": {
                "schema_version": schema_version_out,
                "mode": mode_out,
                "text_prompts": text_prompts if text_prompts else None,
                "positives": portable_positives if portable_positives else None,
                "steps": portable_steps,
                "negatives": portable_negatives,
                "clip_head": clip_head_cfg_clean,
                "optimizer": optimizer_clean,
                "summary": recipe_body.get("summary") or recipe.get("summary"),
            },
        }
        path = (recipes_root / f"{recipe_id}.json").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path_is_within_root_fn(path, recipes_root.resolve()):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_path_invalid")
        try:
            with path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
            # Persist a portable zip alongside the JSON for download.
            zip_path = (recipes_root / f"{recipe_id}.zip").resolve()
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("recipe.json", json.dumps(payload, ensure_ascii=False, indent=2))
                if crops_dir.exists():
                    for crop_file in crops_dir.glob("*.png"):
                        try:
                            zf.write(crop_file, arcname=f"crops/{crop_file.name}")
                        except Exception:
                            continue
                clip_dir = recipe_dir / "clip_head"
                if clip_dir.exists():
                    for artifact in clip_dir.iterdir():
                        try:
                            if not artifact.is_file():
                                continue
                            if artifact.name not in {"head.npz", "meta.json"}:
                                continue
                            zf.write(artifact, arcname=f"clip_head/{artifact.name}")
                        except Exception:
                            continue
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"agent_recipe_save_failed:{exc}",
            ) from exc
        payload["_path"] = str(path)
        payload["_zip"] = str((recipes_root / f"{recipe_id}.zip").resolve())
        cleanup_recipe_dir = False
        return payload
    finally:
        if cleanup_recipe_dir:
            try:
                shutil.rmtree(recipe_dir, ignore_errors=True)
            except Exception:
                pass


def _load_agent_recipe_impl(
    recipe_id: str,
    *,
    recipes_root: Path,
    path_is_within_root_fn,
) -> Dict[str, Any]:
    path = (recipes_root / f"{recipe_id}.json").resolve()
    if not path_is_within_root_fn(path, recipes_root.resolve()) or not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_recipe_not_found")
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if not isinstance(data.get("recipe"), dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        _validate_agent_recipe_structure(data.get("recipe") or {})
        data["_path"] = str(path)
        zip_path = (recipes_root / f"{recipe_id}.zip").resolve()
        if zip_path.exists():
            data["_zip"] = str(zip_path)
        # Inline a small number of crop previews if present on disk (kept small so
        # /agent_mining/apply_image payloads don't explode when the UI forwards recipes).
        recipe_block = data.get("recipe") or {}
        recipe_dir = (recipes_root / recipe_id).resolve()
        if not path_is_within_root_fn(recipe_dir, recipes_root.resolve()):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_path_invalid")
        crop_dir = (recipe_dir / "crops").resolve()
        if not path_is_within_root_fn(crop_dir, recipe_dir):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_path_invalid")
        max_inline = 8
        inlined = 0

        def _inline_crop(entry: Dict[str, Any]) -> None:
            nonlocal inlined
            if inlined >= max_inline:
                return
            crop_path = entry.get("crop_path")
            if not crop_path or not isinstance(crop_path, str):
                return
            try:
                crop_name = Path(crop_path).name
            except Exception:
                crop_name = ""
            abs_path = (crop_dir / crop_name).resolve() if crop_name else None
            if abs_path and path_is_within_root_fn(abs_path, crop_dir) and abs_path.exists() and abs_path.is_file():
                try:
                    with abs_path.open("rb") as cfp:
                        b64 = base64.b64encode(cfp.read()).decode("ascii")
                    entry["crop_base64"] = f"data:image/png;base64,{b64}"
                    entry["crop_path"] = str(Path("crops") / crop_name)
                    inlined += 1
                except Exception:
                    return
            else:
                # Normalize to relative path in case it was absolute originally.
                try:
                    entry["crop_path"] = f"crops/{Path(crop_path).name}"
                except Exception:
                    return

        steps = recipe_block.get("steps") or []
        if isinstance(steps, list):
            for step in steps:
                ex = step.get("exemplar") if isinstance(step, dict) else None
                if isinstance(ex, dict):
                    _inline_crop(ex)
        positives = recipe_block.get("positives") or []
        if isinstance(positives, list):
            for ex in positives:
                if isinstance(ex, dict):
                    _inline_crop(ex)
        negatives = recipe_block.get("negatives") or []
        if isinstance(negatives, list):
            for ex in negatives:
                if isinstance(ex, dict):
                    _inline_crop(ex)
        return data
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_load_failed:{exc}") from exc


def _load_agent_recipe_json_only_impl(
    recipe_id: str,
    *,
    recipes_root: Path,
    path_is_within_root_fn,
) -> Dict[str, Any]:
    """Load an agent recipe payload without inlining crop_base64 blobs (suitable for inference/export)."""
    path = (recipes_root / f"{recipe_id}.json").resolve()
    if not path_is_within_root_fn(path, recipes_root.resolve()) or not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_recipe_not_found")
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        data["_path"] = str(path)
        zip_path = (recipes_root / f"{recipe_id}.zip").resolve()
        if zip_path.exists():
            data["_zip"] = str(zip_path)
        return data
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_load_failed:{exc}") from exc


def _ensure_recipe_zip_impl(
    recipe: Dict[str, Any],
    *,
    recipes_root: Path,
) -> Path:
    recipe_id = recipe.get("id")
    if not recipe_id:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_missing_id")
    zip_path = (recipes_root / f"{recipe_id}.zip").resolve()
    if zip_path.exists():
        return zip_path
    recipe_dir = recipes_root / recipe_id
    crops_dir = recipe_dir / "crops"
    clip_head_dir = recipe_dir / "clip_head"
    try:
        # Never embed crop_base64 blobs inside the portable zip JSON; the PNGs are included separately.
        def _strip_unportable_fields(obj: Any) -> None:
            if isinstance(obj, dict):
                # UI-only / internal fields should not ship in portable zips.
                for k in list(obj.keys()):
                    if isinstance(k, str) and k.startswith("_"):
                        obj.pop(k, None)
                obj.pop("crop_base64", None)
                for v in obj.values():
                    _strip_unportable_fields(v)
            elif isinstance(obj, list):
                for v in obj:
                    _strip_unportable_fields(v)

        clean_recipe = json.loads(json.dumps(recipe))
        _strip_unportable_fields(clean_recipe)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("recipe.json", json.dumps(clean_recipe, ensure_ascii=False, indent=2))
            if crops_dir.exists():
                for crop_file in crops_dir.glob("*.png"):
                    try:
                        zf.write(crop_file, arcname=f"crops/{crop_file.name}")
                    except Exception:
                        continue
            if clip_head_dir.exists():
                for artifact in clip_head_dir.iterdir():
                    try:
                        if not artifact.is_file():
                            continue
                        if artifact.name not in {"head.npz", "meta.json"}:
                            continue
                        zf.write(artifact, arcname=f"clip_head/{artifact.name}")
                    except Exception:
                        continue
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_zip_failed:{exc}") from exc
    return zip_path


def _import_agent_recipe_zip_bytes_impl(
    zip_bytes: bytes,
    *,
    recipes_root: Path,
    max_json_bytes: int,
    max_clip_head_bytes: int,
    max_crops: int,
    max_crop_bytes: int,
    persist_recipe_fn,
) -> Tuple[Optional[str], Dict[str, Any]]:
    data: Dict[str, Any] = {}
    crops: Dict[str, bytes] = {}
    clip_head_files: Dict[str, bytes] = {}
    if not zip_bytes:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_zip_only")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            json_name = None
            for name in names:
                if name.lower().endswith(".json"):
                    json_name = name
                    break
            if not json_name:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_no_json")
            json_info = zf.getinfo(json_name)
            if json_info.file_size > max_json_bytes:
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="agent_recipe_import_json_too_large")
            json_path = Path(json_name)
            if json_path.is_absolute() or ".." in json_path.parts:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_invalid_path")
            with zf.open(json_name) as jf:
                data = json.load(jf)

            total_bytes = 0
            crop_count = 0
            clip_head_bytes = 0
            for name in names:
                info = zf.getinfo(name)
                arc_path = Path(name)
                if arc_path.is_dir():
                    continue
                if arc_path.is_absolute() or ".." in arc_path.parts:
                    raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_invalid_path")
                if len(arc_path.parts) < 2 or arc_path.parts[0] != "crops":
                    # Non-crop artifacts we support (portable CLIP head).
                    if len(arc_path.parts) == 2 and arc_path.parts[0] == "clip_head" and arc_path.name in {"head.npz", "meta.json"}:
                        clip_head_bytes += info.file_size
                        if clip_head_bytes > max_clip_head_bytes:
                            raise HTTPException(
                                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="agent_recipe_import_clip_head_too_large"
                            )
                        clip_head_files[f"clip_head/{arc_path.name}"] = zf.read(name)
                    continue
                if arc_path.suffix.lower() != ".png":
                    continue
                crop_count += 1
                total_bytes += info.file_size
                if crop_count > max_crops or total_bytes > max_crop_bytes:
                    raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="agent_recipe_import_crops_too_large")
                crops[f"crops/{arc_path.name}"] = zf.read(name)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_recipe_import_failed:{exc}") from exc
    if not isinstance(data, dict) or not data:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_no_json")
    dataset_id = data.get("dataset_id") or (data.get("recipe") or {}).get("dataset_id") or ""
    label = data.get("label") or (data.get("recipe") or {}).get("label") or "imported_recipe"
    class_id = data.get("class_id")
    class_name = data.get("class_name")
    meta_overrides = {
        "dataset_signature": data.get("dataset_signature"),
        "labelmap_hash": data.get("labelmap_hash"),
        "labelmap": data.get("labelmap"),
    }
    persisted = persist_recipe_fn(
        dataset_id,
        class_id,
        class_name,
        label,
        data,
        crop_overrides=crops,
        clip_head_overrides=clip_head_files,
        meta_overrides=meta_overrides,
    )
    old_id = data.get("id") if isinstance(data.get("id"), str) else None
    return old_id, persisted


def _prepass_recipe_dir_impl(
    recipe_id: str,
    *,
    create: bool,
    recipes_root: Path,
    sanitize_id_fn,
) -> Path:
    safe = sanitize_id_fn(recipe_id)
    path = recipes_root / safe
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256_path_impl(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tree_filtered_impl(
    src: Path,
    dest: Path,
    *,
    keep_files: Optional[set[str]] = None,
    sha256_fn=_sha256_path_impl,
) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    if not src.exists():
        return copied
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            sub_dest = dest / item.name
            copied.extend(_copy_tree_filtered_impl(item, sub_dest, keep_files=keep_files, sha256_fn=sha256_fn))
            continue
        if keep_files is not None and item.name not in keep_files:
            continue
        target = dest / item.name
        shutil.copy2(item, target)
        copied.append(
            {
                "path": str(target.relative_to(dest.parent)),
                "size": target.stat().st_size,
                "sha256": sha256_fn(target),
            }
        )
    return copied


def _list_prepass_recipes_impl(
    *,
    recipes_root: Path,
    meta_filename: str,
) -> List[Dict[str, Any]]:
    recipes: List[Dict[str, Any]] = []
    for entry in recipes_root.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / meta_filename
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        recipes.append(
            {
                "id": meta.get("id") or entry.name,
                "name": meta.get("name") or entry.name,
                "description": meta.get("description") or "",
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
            }
        )
    recipes.sort(key=lambda r: r.get("updated_at") or r.get("created_at") or 0, reverse=True)
    return recipes


def _collect_recipe_assets_impl(
    recipe_meta: Dict[str, Any],
    temp_dir: Path,
    *,
    read_labelmap_lines_fn,
    load_labelmap_meta_fn,
    active_labelmap_path: Optional[str],
    sanitize_run_id_fn,
    copy_tree_filtered_fn,
    sha256_fn,
    get_qwen_model_entry_fn,
    resolve_classifier_path_fn,
    yolo_job_root: Path,
    rfdetr_job_root: Path,
    rfdetr_keep_files: Optional[set[str]],
    qwen_metadata_filename: str,
    qwen_job_root: Path,
    upload_root: Path,
    calibration_root: Path,
) -> Dict[str, Any]:
    assets: Dict[str, Any] = {"copied": [], "missing": []}
    config = recipe_meta.get("config") or {}
    glossary_text = recipe_meta.get("glossary") or ""
    if glossary_text:
        glossary_path = temp_dir / "glossary.json"
        glossary_path.write_text(json.dumps({"glossary": glossary_text}, indent=2), encoding="utf-8")
        assets["copied"].append(
            {
                "path": "glossary.json",
                "size": glossary_path.stat().st_size,
                "sha256": sha256_fn(glossary_path),
            }
        )

    labelmap_lines: List[str] = []
    if isinstance(config.get("labelmap"), list):
        labelmap_lines = [str(x).strip() for x in config.get("labelmap") or [] if str(x).strip()]
    if not labelmap_lines:
        dataset_id = config.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id.strip():
            labelmap_lines, _ = load_labelmap_meta_fn(dataset_id)
    if not labelmap_lines and active_labelmap_path:
        try:
            labelmap_lines = read_labelmap_lines_fn(Path(active_labelmap_path))
        except Exception:
            labelmap_lines = []
    if labelmap_lines:
        labelmap_path = temp_dir / "labelmap.txt"
        labelmap_path.write_text("\n".join(labelmap_lines) + "\n", encoding="utf-8")
        assets["copied"].append(
            {
                "path": "labelmap.txt",
                "size": labelmap_path.stat().st_size,
                "sha256": sha256_fn(labelmap_path),
            }
        )

    def _copy_run(root: Path, run_id: Optional[str], keep: Optional[set[str]], kind: str):
        if not run_id:
            return
        run_dir = root / sanitize_run_id_fn(run_id)
        if not run_dir.exists():
            assets["missing"].append({"kind": kind, "id": run_id})
            return
        dest = temp_dir / "models" / kind / run_dir.name
        assets["copied"].extend(copy_tree_filtered_fn(run_dir, dest, keep_files=keep))

    _copy_run(yolo_job_root, config.get("yolo_id"), None, "yolo_runs")
    _copy_run(rfdetr_job_root, config.get("rfdetr_id"), rfdetr_keep_files, "rfdetr_runs")

    copied_qwen_ids: set[str] = set()

    def _copy_qwen_run(model_id: Optional[str]) -> None:
        if not model_id:
            return
        if model_id in copied_qwen_ids:
            return
        entry = get_qwen_model_entry_fn(str(model_id))
        if not entry:
            assets["missing"].append({"kind": "qwen_model", "id": model_id})
            return
        raw_path = entry.get("path")
        if not raw_path:
            assets["missing"].append({"kind": "qwen_model", "id": model_id})
            return
        run_path = Path(str(raw_path)).resolve()
        if run_path.name == "latest":
            run_path = run_path.parent
        if not run_path.exists():
            assets["missing"].append({"kind": "qwen_model", "id": model_id})
            return
        dest = temp_dir / "models" / "qwen_runs" / run_path.name
        assets["copied"].extend(copy_tree_filtered_fn(run_path, dest, keep_files=None))
        copied_qwen_ids.add(model_id)

    _copy_qwen_run(config.get("model_id"))
    _copy_qwen_run(config.get("prepass_caption_model_id"))

    classifier_id = config.get("classifier_id")
    if classifier_id:
        try:
            classifier_path = resolve_classifier_path_fn(classifier_id)
        except HTTPException:
            classifier_path = None
        if classifier_path and classifier_path.exists():
            dest = temp_dir / "models" / "classifiers"
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / classifier_path.name
            shutil.copy2(classifier_path, target)
            assets["copied"].append(
                {
                    "path": str(target.relative_to(temp_dir)),
                    "size": target.stat().st_size,
                    "sha256": sha256_fn(target),
                }
            )
        else:
            assets["missing"].append({"kind": "classifier", "id": classifier_id})

    job_id = config.get("ensemble_job_id")
    if job_id:
        job_dir = calibration_root / sanitize_run_id_fn(job_id)
        if job_dir.exists():
            dest = temp_dir / "models" / "calibration_jobs" / job_dir.name
            assets["copied"].extend(copy_tree_filtered_fn(job_dir, dest, keep_files=None))
        else:
            assets["missing"].append({"kind": "calibration_job", "id": job_id})

    return assets


def _import_prepass_recipe_from_zip_impl(
    zip_path: Path,
    *,
    prepass_recipe_meta: str,
    prepass_schema_version: int,
    prepass_recipe_root: Path,
    prepass_tmp_root: Path,
    yolo_job_root: Path,
    rfdetr_job_root: Path,
    rfdetr_keep_files: Optional[set[str]],
    qwen_job_root: Path,
    qwen_metadata_filename: str,
    upload_root: Path,
    calibration_root: Path,
    read_labelmap_lines_fn,
    validate_manifest_fn,
    unique_name_fn,
    normalize_glossary_fn,
    write_meta_fn,
    sanitize_run_id_fn,
) -> Dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix="prepass_recipe_import_"))
    try:
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        manifest_path = extract_dir / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prepass_recipe_missing_manifest")
        manifest = json.loads(manifest_path.read_text())
        meta_path = extract_dir / prepass_recipe_meta
        if not meta_path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prepass_recipe_missing_meta")
        meta = json.loads(meta_path.read_text())
        validate_manifest_fn(manifest, extract_dir)
        config = meta.get("config") or {}
        if isinstance(config, dict):
            config = dict(config)
            config.pop("dataset_id", None)
        glossary = meta.get("glossary") or ""
        labelmap_file = None
        for candidate in (extract_dir / "labelmap.txt", extract_dir / "labelmaps" / "labelmap.txt"):
            if candidate.exists():
                labelmap_file = candidate
                break
        if labelmap_file and not isinstance(config.get("labelmap"), list):
            try:
                lines = read_labelmap_lines_fn(labelmap_file)
            except Exception:
                lines = []
            if lines:
                config["labelmap"] = lines

        def _run_dir_matches(src: Path, dest: Path, keep_files: Optional[set[str]] = None) -> bool:
            if not dest.exists() or not dest.is_dir():
                return False
            for item in src.iterdir():
                if not item.is_file():
                    continue
                if keep_files is not None and item.name not in keep_files:
                    continue
                target = dest / item.name
                if not target.exists():
                    return False
                if target.stat().st_size != item.stat().st_size:
                    return False
            return True

        def _copy_run_assets(kind: str, root: Path, keep_files: Optional[set[str]] = None) -> Optional[str]:
            src_root = extract_dir / "models" / kind
            if not src_root.exists():
                return None
            for run_dir in src_root.iterdir():
                if not run_dir.is_dir():
                    continue
                existing = root / run_dir.name
                if _run_dir_matches(run_dir, existing, keep_files=keep_files):
                    return run_dir.name
                new_id = uuid.uuid4().hex
                dest = root / new_id
                dest.mkdir(parents=True, exist_ok=True)
                for item in run_dir.iterdir():
                    if item.is_file():
                        if keep_files is not None and item.name not in keep_files:
                            continue
                        shutil.copy2(item, dest / item.name)
                return new_id
            return None

        yolo_id = _copy_run_assets("yolo_runs", yolo_job_root, keep_files=None)
        rfdetr_id = _copy_run_assets("rfdetr_runs", rfdetr_job_root, keep_files=rfdetr_keep_files)
        if yolo_id:
            config["yolo_id"] = yolo_id
        if rfdetr_id:
            config["rfdetr_id"] = rfdetr_id

        qwen_id_map: Dict[str, str] = {}
        qwen_root = extract_dir / "models" / "qwen_runs"
        if qwen_root.exists():
            for run_dir in qwen_root.iterdir():
                if not run_dir.is_dir():
                    continue
                meta_file = run_dir / qwen_metadata_filename
                meta_payload: Dict[str, Any] = {}
                if meta_file.exists():
                    try:
                        meta_payload = json.loads(meta_file.read_text())
                    except Exception:
                        meta_payload = {}
                old_id = str(meta_payload.get("id") or run_dir.name)
                new_id = uuid.uuid4().hex
                dest = qwen_job_root / new_id
                dest.mkdir(parents=True, exist_ok=True)
                for item in run_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dest / item.name)
                # Update metadata id to match new run id if we can.
                meta_dest = dest / qwen_metadata_filename
                if meta_dest.exists():
                    try:
                        payload = json.loads(meta_dest.read_text())
                    except Exception:
                        payload = {}
                    payload["id"] = new_id
                    meta_dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                qwen_id_map[old_id] = new_id

        if qwen_id_map:
            for key in ("model_id", "prepass_caption_model_id"):
                val = config.get(key)
                if isinstance(val, str) and val in qwen_id_map:
                    config[key] = qwen_id_map[val]

        classifier_root = extract_dir / "models" / "classifiers"
        if classifier_root.exists():
            classifier_root.mkdir(parents=True, exist_ok=True)
            for item in classifier_root.iterdir():
                if item.is_file():
                    dest = (upload_root / "classifiers") / item.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists() or dest.stat().st_size != item.stat().st_size:
                        shutil.copy2(item, dest)
                    config["classifier_id"] = str(dest.relative_to(upload_root / "classifiers"))
                    break

        calib_root = extract_dir / "models" / "calibration_jobs"
        if calib_root.exists():
            for job_dir in calib_root.iterdir():
                if not job_dir.is_dir():
                    continue
                existing = calibration_root / job_dir.name
                if _run_dir_matches(job_dir, existing, keep_files=None):
                    config["ensemble_job_id"] = job_dir.name
                else:
                    new_job = uuid.uuid4().hex
                    dest = calibration_root / new_job
                    dest.mkdir(parents=True, exist_ok=True)
                    for item in job_dir.iterdir():
                        if item.is_file():
                            shutil.copy2(item, dest / item.name)
                    config["ensemble_job_id"] = new_job
                break

        original_name = meta.get("name") or f"Imported recipe {uuid.uuid4().hex[:8]}"
        unique_name, renamed_from = unique_name_fn(original_name)
        notice = None
        if renamed_from:
            notice = f"Recipe name '{renamed_from}' already exists. Imported as '{unique_name}'."
        recipe_id = uuid.uuid4().hex
        recipe_dir = _prepass_recipe_dir_impl(
            recipe_id,
            create=True,
            recipes_root=prepass_recipe_root,
            sanitize_id_fn=sanitize_run_id_fn,
        )
        now = time.time()
        recipe_meta = {
            "id": recipe_id,
            "schema_version": prepass_schema_version,
            "name": unique_name,
            "description": meta.get("description") or "",
            "config": config,
            "glossary": normalize_glossary_fn(glossary),
            "created_at": now,
            "updated_at": now,
        }
        write_meta_fn(recipe_dir, recipe_meta)
        return {
            "id": recipe_id,
            "name": recipe_meta["name"],
            "description": recipe_meta.get("description"),
            "created_at": now,
            "updated_at": now,
            "config": recipe_meta["config"],
            "glossary": recipe_meta.get("glossary") or None,
            "renamed_from": renamed_from,
            "notice": notice,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _export_prepass_recipe_impl(
    recipe_id: str,
    *,
    prepass_recipe_meta: str,
    prepass_schema_version: int,
    prepass_recipe_export_root: Path,
    prepass_recipe_root: Path,
    sanitize_run_id_fn,
    load_meta_fn,
    collect_assets_fn,
) -> Path:
    recipe_dir = _prepass_recipe_dir_impl(
        recipe_id,
        create=False,
        recipes_root=prepass_recipe_root,
        sanitize_id_fn=sanitize_run_id_fn,
    )
    meta = load_meta_fn(recipe_dir)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"prepass_recipe_{recipe_id}_", dir=prepass_recipe_export_root)
    )
    meta_copy = json.loads(json.dumps(meta))
    config_copy = meta_copy.get("config") or {}
    if isinstance(config_copy, dict) and "dataset_id" in config_copy:
        config_copy = dict(config_copy)
        config_copy.pop("dataset_id", None)
        meta_copy["config"] = config_copy
    meta_path = temp_dir / prepass_recipe_meta
    meta_path.write_text(json.dumps(meta_copy, indent=2), encoding="utf-8")
    assets = collect_assets_fn(meta_copy, temp_dir)
    manifest = {
        "schema_version": prepass_schema_version,
        "recipe_id": meta.get("id") or recipe_id,
        "generated_at": time.time(),
        "assets": assets,
    }
    manifest_path = temp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = temp_dir.with_suffix(".zip")
    shutil.make_archive(zip_path.with_suffix("").as_posix(), "zip", temp_dir.as_posix())
    return zip_path


def _save_prepass_recipe_impl(
    payload: Dict[str, Any],
    *,
    recipe_id: str,
    prepass_schema_version: int,
    recipes_root: Path,
    sanitize_run_id_fn,
    normalize_glossary_fn,
    write_meta_fn,
) -> Dict[str, Any]:
    recipe_dir = _prepass_recipe_dir_impl(
        recipe_id,
        create=True,
        recipes_root=recipes_root,
        sanitize_id_fn=sanitize_run_id_fn,
    )
    now = time.time()
    existing = {}
    meta_path = recipe_dir / "recipe.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
        except Exception:
            existing = {}
    created_at = float(existing.get("created_at") or now)
    recipe_meta = {
        "id": recipe_id,
        "schema_version": prepass_schema_version,
        "name": str(payload.get("name") or "").strip(),
        "description": (payload.get("description") or "").strip(),
        "config": payload.get("config") or {},
        "glossary": normalize_glossary_fn(payload.get("glossary")),
        "created_at": created_at,
        "updated_at": now,
    }
    write_meta_fn(recipe_dir, recipe_meta)
    return {
        "id": recipe_id,
        "name": recipe_meta["name"],
        "description": recipe_meta.get("description"),
        "created_at": created_at,
        "updated_at": now,
        "config": recipe_meta["config"],
        "glossary": recipe_meta.get("glossary") or None,
    }


def _delete_prepass_recipe_impl(
    recipe_id: str,
    *,
    recipes_root: Path,
    sanitize_run_id_fn,
) -> None:
    recipe_dir = _prepass_recipe_dir_impl(
        recipe_id,
        create=False,
        recipes_root=recipes_root,
        sanitize_id_fn=sanitize_run_id_fn,
    )
    if not recipe_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prepass_recipe_not_found")
    shutil.rmtree(recipe_dir, ignore_errors=True)


def _get_prepass_recipe_impl(
    recipe_id: str,
    *,
    recipes_root: Path,
    sanitize_run_id_fn,
    load_meta_fn,
    prepass_schema_version: int,
) -> Dict[str, Any]:
    recipe_dir = _prepass_recipe_dir_impl(
        recipe_id,
        create=False,
        recipes_root=recipes_root,
        sanitize_id_fn=sanitize_run_id_fn,
    )
    meta = load_meta_fn(recipe_dir)
    return {
        "id": meta.get("id") or recipe_id,
        "name": meta.get("name") or recipe_id,
        "description": meta.get("description"),
        "created_at": float(meta.get("created_at") or time.time()),
        "updated_at": float(meta.get("updated_at") or time.time()),
        "config": meta.get("config") or {},
        "glossary": meta.get("glossary"),
        "schema_version": int(meta.get("schema_version") or prepass_schema_version),
    }
