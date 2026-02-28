"""Agent cascade persistence helpers."""

from __future__ import annotations

import io
import json
import shutil
import stat
import tempfile
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
    HTTP_413_CONTENT_TOO_LARGE,
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
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if zf.testzip() is None:
                    return zip_path
        except Exception:
            try:
                zip_path.unlink()
            except Exception:
                pass

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

    def _safe_archive_relpath(raw: str) -> Optional[str]:
        rel = str(raw or "").replace("\\", "/").strip().strip("/")
        if not rel:
            return None
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            return None
        return rel

    # Build zip bundle.
    temp_zip_path = zip_path.with_suffix(f"{zip_path.suffix}.{uuid.uuid4().hex}.tmp")
    try:
        clean_cascade = json.loads(json.dumps(cascade))
        for key in list(clean_cascade.keys()):
            if isinstance(key, str) and key.startswith("_"):
                clean_cascade.pop(key, None)
        with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("cascade.json", json.dumps(clean_cascade, ensure_ascii=False, indent=2))

            # Attach recipe zips.
            for rid in sorted(set(recipe_ids)):
                if not rid:
                    continue
                safe_recipe_name = _safe_archive_relpath(f"{rid}.zip")
                if not safe_recipe_name:
                    continue
                try:
                    recipe_obj = load_recipe_fn(rid)
                except Exception:
                    recipe_obj = {"id": rid}
                recipe_zip = ensure_recipe_zip_fn(recipe_obj)
                try:
                    recipe_zip_resolved = recipe_zip.resolve()
                except Exception:
                    continue
                if not path_is_within_root_fn(recipe_zip_resolved, recipes_root.resolve()):
                    continue
                if recipe_zip.exists():
                    try:
                        zf.write(recipe_zip, arcname=f"recipes/{safe_recipe_name}")
                    except Exception:
                        continue

            # Attach classifier artifacts referenced by steps.
            for rel_path in sorted(set(classifier_paths)):
                safe_classifier_rel = _safe_archive_relpath(rel_path)
                if not safe_classifier_rel:
                    continue
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
                    arcname = f"classifiers/{safe_classifier_rel}"
                    try:
                        zf.write(resolved_path, arcname=arcname)
                    except Exception:
                        pass
                meta_path = resolved_path.with_suffix(resolved_path.suffix + ".meta.pkl")
                if meta_path.exists() and meta_path.is_file():
                    try:
                        meta_rel = f"classifiers/{safe_classifier_rel}.meta.pkl"
                        zf.write(meta_path, arcname=meta_rel)
                    except Exception:
                        pass
        temp_zip_path.replace(zip_path)
    except Exception as exc:  # noqa: BLE001
        try:
            temp_zip_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_cascade_zip_failed:{exc}") from exc
    return zip_path


def _import_agent_cascade_zip_obj_impl(
    zf: zipfile.ZipFile,
    *,
    cascades_root: Path,
    classifiers_root: Path,
    max_json_bytes: int,
    classifier_allowed_exts: List[str],
    path_is_within_root_fn,
    import_recipe_fn=None,
    import_recipe_file_fn=None,
    persist_cascade_fn,
    max_entry_bytes: Optional[int] = None,
    max_recipe_zip_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    names = zf.namelist()
    cascade_name = None
    for name in names:
        if Path(name).name.lower() == "cascade.json":
            cascade_name = name
            break
    if not cascade_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_no_json")
    info = zf.getinfo(cascade_name)
    if max_entry_bytes and info.file_size > max_entry_bytes:
        raise HTTPException(
            status_code=HTTP_413_CONTENT_TOO_LARGE,
            detail="agent_cascade_import_entry_too_large",
        )
    if info.file_size > max_json_bytes:
        raise HTTPException(status_code=HTTP_413_CONTENT_TOO_LARGE, detail="agent_cascade_import_json_too_large")
    cascade_path = Path(cascade_name)
    if cascade_path.is_absolute() or ".." in cascade_path.parts:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_invalid_path")
    with zf.open(cascade_name) as jf:
        cascade_data = json.load(jf)
    if not isinstance(cascade_data, dict):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_invalid_schema")

    recipe_zip_names: List[str] = []
    classifier_file_names: List[str] = []
    for name in names:
        arc_path = Path(name)
        if arc_path.is_dir():
            continue
        if arc_path.is_absolute() or ".." in arc_path.parts:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_invalid_path")
        try:
            member_info = zf.getinfo(name)
        except KeyError:
            continue
        mode = (member_info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_symlink_unsupported")
        if max_entry_bytes and member_info.file_size > max_entry_bytes:
            raise HTTPException(
                status_code=HTTP_413_CONTENT_TOO_LARGE,
                detail="agent_cascade_import_entry_too_large",
            )
        if len(arc_path.parts) >= 2 and arc_path.parts[0] == "recipes" and arc_path.suffix.lower() == ".zip":
            if max_recipe_zip_bytes and member_info.file_size > max_recipe_zip_bytes:
                raise HTTPException(
                    status_code=HTTP_413_CONTENT_TOO_LARGE,
                    detail="agent_cascade_import_recipe_too_large",
                )
            recipe_zip_names.append(name)
        if len(arc_path.parts) >= 2 and arc_path.parts[0] == "classifiers":
            if arc_path.name.endswith(".meta.pkl") or arc_path.suffix.lower() in classifier_allowed_exts:
                classifier_file_names.append(name)

    classifier_map: Dict[str, str] = {}
    if classifier_file_names:
        import_tag = f"cascade_{uuid.uuid4().hex[:8]}"
        import_root = (classifiers_root / "imports" / import_tag).resolve()
        if not path_is_within_root_fn(import_root, classifiers_root.resolve()):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_invalid_path")
        import_root.mkdir(parents=True, exist_ok=True)
        for name in classifier_file_names:
            arc_path = Path(name)
            rel_inside = Path(*arc_path.parts[1:])
            dest_path = (import_root / rel_inside).resolve()
            if not path_is_within_root_fn(dest_path, classifiers_root.resolve()):
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_invalid_path")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(name) as src, dest_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            except Exception:
                continue
            if not arc_path.name.endswith(".meta.pkl") and arc_path.suffix.lower() in classifier_allowed_exts:
                orig_rel = str(rel_inside.as_posix())
                new_rel = str((Path("imports") / import_tag / rel_inside).as_posix())
                classifier_map[orig_rel] = new_rel

    id_map: Dict[str, str] = {}
    recipe_import_tmp_root: Optional[Path] = None
    try:
        for name in recipe_zip_names:
            src_id = None
            try:
                arc_path = Path(name)
                src_id = arc_path.stem
            except Exception:
                src_id = None
            if import_recipe_file_fn is not None:
                if recipe_import_tmp_root is None:
                    recipe_import_tmp_root = Path(tempfile.mkdtemp(prefix="agent_cascade_recipe_import_"))
                recipe_zip_path = recipe_import_tmp_root / f"{uuid.uuid4().hex}.zip"
                # Stream recipe archives to disk so cascade import does not materialize
                # large nested zips in RAM before handing off to recipe importer.
                with zf.open(name) as src, recipe_zip_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                old_id, persisted = import_recipe_file_fn(recipe_zip_path)
                try:
                    recipe_zip_path.unlink()
                except Exception:
                    pass
            elif import_recipe_fn is not None:
                old_id, persisted = import_recipe_fn(zf.read(name))
            else:
                raise HTTPException(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="agent_cascade_import_recipe_handler_missing",
                )
            new_id = persisted.get("id")
            if isinstance(new_id, str) and new_id:
                if old_id:
                    id_map[str(old_id)] = str(new_id)
                if src_id:
                    id_map[str(src_id)] = str(new_id)
    finally:
        if recipe_import_tmp_root is not None:
            shutil.rmtree(recipe_import_tmp_root, ignore_errors=True)

    label = cascade_data.get("label") or "imported_cascade"
    steps_raw = cascade_data.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_invalid_schema")
    dedupe_raw = cascade_data.get("dedupe") if isinstance(cascade_data.get("dedupe"), dict) else {}

    steps_out: List[Dict[str, Any]] = []
    for raw in steps_raw:
        if not isinstance(raw, dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_invalid_schema")
        rid = raw.get("recipe_id")
        if not rid and isinstance(raw.get("recipe"), dict):
            rid = raw["recipe"].get("id")
        if not isinstance(rid, str) or not rid.strip():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_invalid_schema")
        rid = rid.strip()
        mapped = id_map.get(rid)
        if not mapped:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_missing_recipe")
        extra_classifier_ref = None
        if isinstance(raw.get("extra_clip_classifier_path"), str) and raw.get("extra_clip_classifier_path").strip():
            extra_classifier_ref = str(raw.get("extra_clip_classifier_path")).strip()
            if classifier_map:
                mapped_classifier = classifier_map.get(extra_classifier_ref)
                if not mapped_classifier:
                    raise HTTPException(
                        status_code=HTTP_400_BAD_REQUEST,
                        detail="agent_cascade_import_missing_classifier",
                    )
                extra_classifier_ref = mapped_classifier
        steps_out.append(
            {
                "enabled": bool(raw.get("enabled", True)),
                "recipe_id": mapped,
                "override_class_id": raw.get("override_class_id"),
                "override_class_name": raw.get("override_class_name"),
                "dedupe_group": raw.get("dedupe_group"),
                "participate_cross_class_dedupe": bool(raw.get("participate_cross_class_dedupe", True)),
                "clip_head_min_prob_override": raw.get("clip_head_min_prob_override"),
                "clip_head_margin_override": raw.get("clip_head_margin_override"),
                "extra_clip_classifier_path": extra_classifier_ref,
                "extra_clip_min_prob": raw.get("extra_clip_min_prob"),
                "extra_clip_margin": raw.get("extra_clip_margin"),
            }
        )
    return persist_cascade_fn(str(label), {"steps": steps_out, "dedupe": dedupe_raw})


def _import_agent_cascade_zip_bytes_impl(
    zip_bytes: bytes,
    *,
    cascades_root: Path,
    classifiers_root: Path,
    max_json_bytes: int,
    classifier_allowed_exts: List[str],
    path_is_within_root_fn,
    import_recipe_fn=None,
    import_recipe_file_fn=None,
    persist_cascade_fn,
    max_entry_bytes: Optional[int] = None,
    max_recipe_zip_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    if not zip_bytes:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_zip_only")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            return _import_agent_cascade_zip_obj_impl(
                zf,
                cascades_root=cascades_root,
                classifiers_root=classifiers_root,
                max_json_bytes=max_json_bytes,
                classifier_allowed_exts=classifier_allowed_exts,
                path_is_within_root_fn=path_is_within_root_fn,
                import_recipe_fn=import_recipe_fn,
                import_recipe_file_fn=import_recipe_file_fn,
                persist_cascade_fn=persist_cascade_fn,
                max_entry_bytes=max_entry_bytes,
                max_recipe_zip_bytes=max_recipe_zip_bytes,
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_cascade_import_failed:{exc}") from exc


def _import_agent_cascade_zip_file_impl(
    zip_path: Path,
    *,
    cascades_root: Path,
    classifiers_root: Path,
    max_json_bytes: int,
    classifier_allowed_exts: List[str],
    path_is_within_root_fn,
    import_recipe_fn=None,
    import_recipe_file_fn=None,
    persist_cascade_fn,
    max_entry_bytes: Optional[int] = None,
    max_recipe_zip_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    if not zip_path.exists() or not zip_path.is_file():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_cascade_import_zip_only")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return _import_agent_cascade_zip_obj_impl(
                zf,
                cascades_root=cascades_root,
                classifiers_root=classifiers_root,
                max_json_bytes=max_json_bytes,
                classifier_allowed_exts=classifier_allowed_exts,
                path_is_within_root_fn=path_is_within_root_fn,
                import_recipe_fn=import_recipe_fn,
                import_recipe_file_fn=import_recipe_file_fn,
                persist_cascade_fn=persist_cascade_fn,
                max_entry_bytes=max_entry_bytes,
                max_recipe_zip_bytes=max_recipe_zip_bytes,
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_cascade_import_failed:{exc}") from exc
