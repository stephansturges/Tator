from __future__ import annotations

import json
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)


def _persist_agent_cascade_impl(
    label: str,
    payload: Dict[str, Any],
    *,
    cascades_root: Path,
    path_is_within_root_fn,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_invalid_schema")
    cascade_id = f"ac_{uuid.uuid4().hex[:8]}"
    record = {
        "id": cascade_id,
        "label": (label or "").strip() or "agent_cascade",
        "created_at": time.time(),
        **payload,
    }
    path = (cascades_root / f"{cascade_id}.json").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path_is_within_root_fn(path, cascades_root.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_path_invalid")
    try:
        with path.open("w", encoding="utf-8") as fp:
            json.dump(record, fp, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_cascade_save_failed:{exc}") from exc
    record["_path"] = str(path)
    zip_path = (cascades_root / f"{cascade_id}.zip").resolve()
    if zip_path.exists():
        record["_zip"] = str(zip_path)
    return record


def _load_agent_cascade_impl(
    cascade_id: str,
    *,
    cascades_root: Path,
    path_is_within_root_fn,
) -> Dict[str, Any]:
    path = (cascades_root / f"{cascade_id}.json").resolve()
    if not path_is_within_root_fn(path, cascades_root.resolve()) or not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_cascade_not_found")
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_invalid_schema")
        data["_path"] = str(path)
        zip_path = (cascades_root / f"{cascade_id}.zip").resolve()
        if zip_path.exists():
            data["_zip"] = str(zip_path)
        return data
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_cascade_load_failed:{exc}") from exc


def _list_agent_cascades_impl(*, cascades_root: Path) -> List[Dict[str, Any]]:
    cascades: List[Dict[str, Any]] = []
    for path in cascades_root.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if not isinstance(data, dict):
                continue
            data["_path"] = str(path)
            zip_path = (cascades_root / f"{data.get('id','')}.zip").resolve()
            if zip_path.exists():
                data["_zip"] = str(zip_path)
            cascades.append(data)
        except Exception:
            continue
    cascades.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return cascades


def _delete_agent_cascade_impl(
    cascade_id: str,
    *,
    cascades_root: Path,
    path_is_within_root_fn,
) -> None:
    json_path = (cascades_root / f"{cascade_id}.json").resolve()
    zip_path = (cascades_root / f"{cascade_id}.zip").resolve()
    if not path_is_within_root_fn(json_path, cascades_root.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_path_invalid")
    removed_any = False
    for path in (json_path, zip_path):
        if path.exists():
            try:
                path.unlink()
                removed_any = True
            except Exception:
                pass
    if not removed_any:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_cascade_not_found")


def _ensure_cascade_zip_impl(
    cascade: Dict[str, Any],
    *,
    cascades_root: Path,
    recipes_root: Path,
    classifiers_root: Path,
    path_is_within_root_fn,
    ensure_recipe_zip_fn,
    load_recipe_fn,
    resolve_classifier_fn,
) -> Path:
    cascade_id = cascade.get("id")
    if not cascade_id:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_missing_id")
    zip_path = (cascades_root / f"{cascade_id}.zip").resolve()
    if zip_path.exists():
        return zip_path

    steps = cascade.get("steps") or []
    recipe_ids: List[str] = []
    classifier_paths: List[str] = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        rid = step.get("recipe_id")
        if isinstance(rid, str) and rid.strip():
            recipe_ids.append(rid.strip())
        extra_classifier = step.get("extra_clip_classifier_path")
        if isinstance(extra_classifier, str) and extra_classifier.strip():
            classifier_paths.append(extra_classifier.strip())
    dedupe = cascade.get("dedupe") if isinstance(cascade.get("dedupe"), dict) else {}
    if isinstance(dedupe, dict):
        clip_head_recipe_id = dedupe.get("clip_head_recipe_id")
        if isinstance(clip_head_recipe_id, str) and clip_head_recipe_id.strip():
            recipe_ids.append(clip_head_recipe_id.strip())

    # Build zip bundle.
    try:
        clean_cascade = json.loads(json.dumps(cascade))
        for key in list(clean_cascade.keys()):
            if isinstance(key, str) and key.startswith("_"):
                clean_cascade.pop(key, None)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("cascade.json", json.dumps(clean_cascade, ensure_ascii=False, indent=2))

            # Attach recipe zips.
            for rid in sorted(set(recipe_ids)):
                if not rid:
                    continue
                try:
                    recipe_obj = load_recipe_fn(rid)
                except Exception:
                    recipe_obj = {"id": rid}
                recipe_zip = ensure_recipe_zip_fn(recipe_obj)
                if recipe_zip.exists():
                    try:
                        zf.write(recipe_zip, arcname=f"recipes/{rid}.zip")
                    except Exception:
                        continue

            # Attach classifier artifacts referenced by steps.
            for rel_path in sorted(set(classifier_paths)):
                resolved = resolve_classifier_fn(rel_path)
                if not resolved:
                    continue
                try:
                    resolved_path = Path(resolved).resolve()
                except Exception:
                    continue
                if not path_is_within_root_fn(resolved_path, classifiers_root.resolve()):
                    continue
                if resolved_path.exists() and resolved_path.is_file():
                    arcname = f"classifiers/{rel_path}"
                    try:
                        zf.write(resolved_path, arcname=arcname)
                    except Exception:
                        pass
                meta_path = resolved_path.with_suffix(resolved_path.suffix + ".meta.pkl")
                if meta_path.exists() and meta_path.is_file():
                    try:
                        meta_rel = f"classifiers/{rel_path}.meta.pkl"
                        zf.write(meta_path, arcname=meta_rel)
                    except Exception:
                        pass
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_cascade_zip_failed:{exc}") from exc
    return zip_path
