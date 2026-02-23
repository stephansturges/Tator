#!/usr/bin/env python3
import json
import sys
import time
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path
import tempfile
import zipfile
import base64


def _get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
        return json.loads(text) if text else {}


def _post_multipart(url: str, fields: dict, files: dict, timeout: int = 60):
    boundary = f"----boundary{int(time.time() * 1000)}"
    body = bytearray()
    for name, value in (fields or {}).items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")
    for name, (filename, content, content_type) in (files or {}).items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n".encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else {}


def _safe(label, fn):
    try:
        return {"ok": True, "result": fn()}
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        return {"ok": False, "error": f"HTTP {exc.code}: {body}"}
    except URLError as exc:
        return {"ok": False, "error": f"URL error: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _safe_allow(label, fn, ok_statuses=None):
    ok_statuses = set(ok_statuses or [])
    try:
        return {"ok": True, "result": fn()}
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        if exc.code in ok_statuses:
            return {"ok": True, "result": f"HTTP {exc.code}: {body}"}
        return {"ok": False, "error": f"HTTP {exc.code}: {body}"}
    except URLError as exc:
        return {"ok": False, "error": f"URL error: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _load_fuzz_image() -> tuple[str, int, int, str]:
    root = Path("tests/fixtures/fuzz_pack")
    manifest = json.loads((root / "manifest.json").read_text())
    image_name = sorted(manifest["images"].keys())[0]
    img_path = root / "images" / image_name
    data = img_path.read_bytes()
    try:
        from PIL import Image
        from io import BytesIO
        with Image.open(BytesIO(data)) as img:
            width, height = img.size
    except Exception:
        width = 256
        height = 256
    b64 = base64.b64encode(data).decode("utf-8")
    return b64, width, height, image_name


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    results = {}

    # Basic GETs (install-check parity)
    for path in (
        "/system/health_summary",
        "/system/storage_check",
        "/system/gpu",
        "/qwen/settings",
        "/qwen/train/jobs",
        "/sam3/train/jobs",
        "/sam3/models/available",
        "/sam3/prompt_helper/presets",
        "/clip/train",
        "/clip/backbones",
        "/clip/labelmaps",
        "/clip/classifiers",
        "/clip/active_model",
        "/yolo/variants",
        "/yolo/train/jobs",
        "/rfdetr/variants",
        "/rfdetr/train/jobs",
        "/calibration/jobs",
        "/agent_mining/cache_size",
        "/qwen/train/cache_size",
        "/sam3/train/cache_size",
        "/detectors/default",
        "/yolo/runs",
        "/rfdetr/runs",
        "/sam3/storage/runs",
        "/sam3/models",
        "/qwen/models",
        "/qwen/status",
        "/datasets",
        "/glossaries",
        "/qwen/datasets",
        "/sam3/datasets",
        "/predictor_settings",
        "/sam_slots",
    ):
        results[f"GET {path}"] = _safe(path, lambda p=path: _get(base + p))

    # Detectors default POST (use existing mode)
    det_default = results.get("GET /detectors/default", {}).get("result") or {}
    mode = det_default.get("mode") or det_default.get("active") or "ensemble"
    results["POST /detectors/default"] = _safe(
        "detectors_default",
        lambda: _post(base + "/detectors/default", {"mode": mode}),
    )

    # YOLO active POST (re-assert current)
    yolo_active = _safe("yolo_active", lambda: _get(base + "/yolo/active"))
    results["GET /yolo/active"] = yolo_active
    if yolo_active.get("ok"):
        run_id = (yolo_active.get("result") or {}).get("run_id")
        if run_id:
            results["POST /yolo/active"] = _safe(
                "yolo_active_post",
                lambda: _post(base + "/yolo/active", {"run_id": run_id}),
            )

    # RF-DETR active POST (re-assert current)
    rfdetr_active = _safe("rfdetr_active", lambda: _get(base + "/rfdetr/active"))
    results["GET /rfdetr/active"] = rfdetr_active
    if rfdetr_active.get("ok"):
        run_id = (rfdetr_active.get("result") or {}).get("run_id")
        if run_id:
            results["POST /rfdetr/active"] = _safe(
                "rfdetr_active_post",
                lambda: _post(base + "/rfdetr/active", {"run_id": run_id}),
            )

    # Clip active model POST (reuse first classifier)
    classifiers_payload = results.get("GET /clip/classifiers", {}).get("result")
    classifiers = None
    if isinstance(classifiers_payload, list):
        classifiers = classifiers_payload
    elif isinstance(classifiers_payload, dict):
        classifiers = classifiers_payload.get("classifiers")
    if classifiers:
        first = classifiers[0]
        if isinstance(first, dict):
            path = first.get("path") or first.get("id")
        else:
            path = first
        if path:
            results["POST /clip/active_model"] = _safe(
                "clip_active_post",
                lambda: _post(base + "/clip/active_model", {"path": path}),
            )

    # YOLO/RFDETR run summaries (if any)
    for key, base_path in (("yolo", "/yolo/runs"), ("rfdetr", "/rfdetr/runs")):
        run_list = results.get(f"GET {base_path}", {}).get("result") or []
        if isinstance(run_list, dict):
            run_list = run_list.get("runs") or []
        if run_list:
            run = run_list[0]
            run_id = run.get("run_id") if isinstance(run, dict) else None
            if run_id:
                results[f"GET {base_path}/{{run_id}}/summary"] = _safe(
                    f"{key}_summary", lambda p=base_path, rid=run_id: _get(base + f"{p}/{rid}/summary")
                )

    # Sam3 storage run summary
    sam3_runs = results.get("GET /sam3/storage/runs", {}).get("result") or []
    if isinstance(sam3_runs, dict):
        sam3_runs = sam3_runs.get("runs") or []
    if sam3_runs:
        run = sam3_runs[0]
        run_id = run.get("run_id") if isinstance(run, dict) else None
        if run_id:
            results["GET /sam3/storage/runs/{run_id}"] = _safe(
                "sam3_run", lambda rid=run_id: _get(base + f"/sam3/storage/runs/{rid}")
            )

    # Predictor settings POST
    predictor_payload = results.get("GET /predictor_settings", {}).get("result") or {}
    max_predictors = predictor_payload.get("max_predictors") or predictor_payload.get("activePredictors") or 1
    results["POST /predictor_settings"] = _safe(
        "predictor_settings_post",
        lambda: _post(base + "/predictor_settings", {"max_predictors": int(max_predictors)}),
    )

    # Qwen settings POST (reuse current trust_remote_code)
    qwen_settings = results.get("GET /qwen/settings", {}).get("result") or {}
    trust_remote_code = bool(qwen_settings.get("trust_remote_code"))
    results["POST /qwen/settings"] = _safe(
        "qwen_settings_post",
        lambda: _post(base + "/qwen/settings", {"trust_remote_code": trust_remote_code}),
    )

    # Cache purge endpoints
    results["POST /agent_mining/cache/purge"] = _safe(
        "agent_mining_cache_purge", lambda: _post(base + "/agent_mining/cache/purge", {})
    )
    results["POST /qwen/train/cache/purge"] = _safe(
        "qwen_train_cache_purge", lambda: _post(base + "/qwen/train/cache/purge", {})
    )
    results["POST /sam3/train/cache/purge"] = _safe(
        "sam3_train_cache_purge", lambda: _post(base + "/sam3/train/cache/purge", {})
    )

    # Glossary CRUD (safe temp entry)
    glossary_name = f"ui_contract_test_{int(time.time())}"
    glossary_body = "light_vehicle: car, sedan"
    results["POST /glossaries"] = _safe(
        "glossary_save", lambda: _post(base + "/glossaries", {"name": glossary_name, "glossary": glossary_body})
    )
    results["GET /glossaries/{name}"] = _safe(
        "glossary_get", lambda: _get(base + f"/glossaries/{glossary_name}")
    )
    results["DELETE /glossaries/{name}"] = _safe(
        "glossary_delete", lambda: urllib.request.urlopen(urllib.request.Request(
            base + f"/glossaries/{glossary_name}", method="DELETE")
        ).read().decode("utf-8")
    )

    # Dataset glossary + text labels
    datasets = results.get("GET /datasets", {}).get("result") or []
    if isinstance(datasets, list) and datasets:
        entry = datasets[0]
        dataset_id = entry.get("id") or entry.get("dataset_id")
        if dataset_id:
            results["GET /datasets/{id}/glossary"] = _safe(
                "dataset_glossary_get", lambda: _get(base + f"/datasets/{dataset_id}/glossary")
            )
            results["POST /datasets/{id}/glossary"] = _safe(
                "dataset_glossary_post",
                lambda: _post(base + f"/datasets/{dataset_id}/glossary", {"glossary": glossary_body}),
            )
            dataset_root = entry.get("dataset_root")
            image_name = None
            if dataset_root:
                root = Path(dataset_root)
                for split in ("train", "val"):
                    img_dir = root / split
                    if not img_dir.exists():
                        img_dir = root / split / "images"
                    if img_dir.exists():
                        for path in img_dir.iterdir():
                            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                                image_name = path.name
                                break
                    if image_name:
                        break
            if image_name:
                results["GET /datasets/{id}/text_labels/{image}"] = _safe(
                    "text_label_get", lambda: _get(base + f"/datasets/{dataset_id}/text_labels/{image_name}")
                )
                results["POST /datasets/{id}/text_labels/{image}"] = _safe(
                    "text_label_post",
                    lambda: _post(
                        base + f"/datasets/{dataset_id}/text_labels/{image_name}", {"caption": "contract test"}
                    ),
                )

    # Dataset upload (tiny synthetic YOLO zip)
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ui_contract_dataset_"))
        root = tmp_dir / "mini_dataset"
        (root / "train" / "images").mkdir(parents=True, exist_ok=True)
        (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (root / "labelmap.txt").write_text("object\n")
        try:
            from PIL import Image
            img = Image.new("RGB", (1, 1), (0, 0, 0))
            img.save(root / "train" / "images" / "img1.png")
        except Exception:
            (root / "train" / "images" / "img1.png").write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
            )
        (root / "train" / "labels" / "img1.txt").write_text("0 0.5 0.5 1 1\n")
        zip_path = tmp_dir / "mini_dataset.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in root.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(root.parent)
                    zf.write(path, arcname=str(rel))
        payload = {
            "dataset_id": f"ui_contract_upload_{int(time.time())}",
            "dataset_type": "bbox",
            "context": "contract test",
        }
        with zip_path.open("rb") as handle:
            upload_resp = _safe(
                "dataset_upload",
                lambda: _post_multipart(
                    base + "/datasets/upload",
                    payload,
                    {"file": (zip_path.name, handle.read(), "application/zip")},
                ),
            )
        results["POST /datasets/upload"] = upload_resp
        uploaded_id = None
        if upload_resp.get("ok"):
            uploaded_id = (upload_resp.get("result") or {}).get("id")
        if uploaded_id:
            results["DELETE /datasets/{uploaded}"] = _safe(
                "dataset_delete",
                lambda: urllib.request.urlopen(
                    urllib.request.Request(base + f"/datasets/{uploaded_id}", method="DELETE")
                ).read().decode("utf-8"),
            )
    except Exception as exc:
        results["POST /datasets/upload"] = {"ok": False, "error": str(exc)}

    # Prepass recipe CRUD (store tiny recipe, export/import, cleanup)
    recipe_id = None
    try:
        recipe_payload = {
            "name": f"ui_contract_recipe_{int(time.time())}",
            "description": "contract test",
            "config": {"enable_yolo": True, "enable_rfdetr": True, "sam3_text": True},
            "glossary": glossary_body,
        }
        results["POST /prepass/recipes"] = _safe(
            "prepass_recipe_save", lambda: _post(base + "/prepass/recipes", recipe_payload)
        )
        recipe_id = (results["POST /prepass/recipes"].get("result") or {}).get("id")
        results["GET /prepass/recipes"] = _safe(
            "prepass_recipe_list", lambda: _get(base + "/prepass/recipes")
        )
        export_bytes = None
        if recipe_id:
            results["GET /prepass/recipes/{id}"] = _safe(
                "prepass_recipe_get", lambda rid=recipe_id: _get(base + f"/prepass/recipes/{rid}")
            )
            # Export recipe zip
            def _export_recipe(rid=recipe_id):
                req = urllib.request.Request(base + f"/prepass/recipes/{rid}/export", method="POST")
                with urllib.request.urlopen(req) as resp:
                    data = resp.read()
                    return data
            results["POST /prepass/recipes/{id}/export"] = _safe(
                "prepass_recipe_export", _export_recipe
            )
            if results["POST /prepass/recipes/{id}/export"].get("ok"):
                export_bytes = results["POST /prepass/recipes/{id}/export"]["result"]
            # Import raw (json body)
            if export_bytes:
                def _import_raw(blob=export_bytes):
                    req = urllib.request.Request(
                        base + "/prepass/recipes/import-raw",
                        data=blob,
                        headers={"Content-Type": "application/zip"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req) as resp:
                        payload = resp.read().decode("utf-8")
                        return json.loads(payload) if payload else {}
                results["POST /prepass/recipes/import-raw"] = _safe(
                    "prepass_recipe_import_raw",
                    _import_raw,
                )
    except Exception as exc:
        results["POST /prepass/recipes"] = {"ok": False, "error": str(exc)}
    finally:
        if recipe_id:
            results["DELETE /prepass/recipes/{id}"] = _safe(
                "prepass_recipe_delete",
                lambda rid=recipe_id: urllib.request.urlopen(
                    urllib.request.Request(base + f"/prepass/recipes/{rid}", method="DELETE")
                ).read().decode("utf-8"),
            )

    # Agent mining recipes list + delete (allow 404)
    results["GET /agent_mining/recipes"] = _safe(
        "agent_mining_recipes", lambda: _get(base + "/agent_mining/recipes")
    )
    results["DELETE /agent_mining/recipes/{id}"] = _safe_allow(
        "agent_mining_recipe_delete",
        lambda: urllib.request.urlopen(
            urllib.request.Request(base + "/agent_mining/recipes/ui_contract_missing", method="DELETE")
        ).read().decode("utf-8"),
        ok_statuses={404},
    )

    # Run deletion endpoints (use missing ids, allow 404/400)
    for path in (
        "/yolo/runs/ui_contract_missing",
        "/rfdetr/runs/ui_contract_missing",
        "/sam3/storage/runs/ui_contract_missing",
    ):
        results[f"DELETE {path}"] = _safe_allow(
            f"delete_{path}",
            lambda p=path: urllib.request.urlopen(
                urllib.request.Request(base + p, method="DELETE")
            ).read().decode("utf-8"),
            ok_statuses={400, 404},
        )

    # Dataset conversion / crop-zip endpoints (missing ids OK)
    results["POST /sam3/datasets/{id}/convert"] = _safe_allow(
        "sam3_dataset_convert",
        lambda: urllib.request.urlopen(
            urllib.request.Request(
                base + "/sam3/datasets/ui_contract_missing/convert",
                method="POST",
            )
        ).read().decode("utf-8"),
        ok_statuses={400, 404},
    )
    results["POST /crop_zip_init"] = _safe_allow(
        "crop_zip_init",
        lambda: _post(
            base + "/crop_zip_init",
            {"dataset_id": "ui_contract_missing", "labelmap": []},
        ),
        ok_statuses={400, 404},
    )
    results["GET /crop_zip_finalize"] = _safe_allow(
        "crop_zip_finalize",
        lambda: _get(
            base + "/crop_zip_finalize?jobId=ui_contract_missing",
        ),
        ok_statuses={400, 404},
    )

    # Clip classifier management (missing ids OK)
    results["POST /clip/classifiers/rename"] = _safe_allow(
        "clip_classifier_rename",
        lambda: _post_multipart(
            base + "/clip/classifiers/rename",
            {"rel_path": "ui_contract_missing.pkl", "new_name": "ui_contract_missing_renamed.pkl"},
            {},
        ),
        ok_statuses={400, 404},
    )
    results["DELETE /clip/classifiers"] = _safe_allow(
        "clip_classifier_delete",
        lambda: urllib.request.urlopen(
            urllib.request.Request(
                base + "/clip/classifiers?rel_path=ui_contract_missing.pkl",
                method="DELETE",
            )
        ).read().decode("utf-8"),
        ok_statuses={400, 404},
    )

    # Cancel endpoints (allow 404/400 if job not found)
    for cancel_path in (
        "/calibration/jobs/ui_contract_missing/cancel",
        "/agent_mining/jobs/ui_contract_missing/cancel",
        "/qwen/train/jobs/ui_contract_missing/cancel",
        "/sam3/train/jobs/ui_contract_missing/cancel",
        "/yolo/train/jobs/ui_contract_missing/cancel",
        "/yolo/head_graft/jobs/ui_contract_missing/cancel",
        "/rfdetr/train/jobs/ui_contract_missing/cancel",
    ):
        results[f"POST {cancel_path}"] = _safe_allow(
            f"cancel_{cancel_path}",
            lambda p=cancel_path: urllib.request.urlopen(
                urllib.request.Request(base + p, method="POST")
            ).read().decode("utf-8"),
            ok_statuses={400, 404},
        )

    # SAM and base64 endpoints
    try:
        img_b64, width, height, image_name = _load_fuzz_image()
        results["POST /predict_base64"] = _safe(
            "predict_base64",
            lambda: _post(base + "/predict_base64", {"image_base64": img_b64}),
        )
        bbox_payload = {
            "image_base64": img_b64,
            "bbox_left": float(width) * 0.1,
            "bbox_top": float(height) * 0.1,
            "bbox_width": float(width) * 0.2,
            "bbox_height": float(height) * 0.2,
            "uuid": "contract",
            "image_name": image_name,
        }
        point_payload = {
            "image_base64": img_b64,
            "point_x": float(width) * 0.5,
            "point_y": float(height) * 0.5,
            "uuid": "contract",
            "image_name": image_name,
        }
        multi_payload = {
            "image_base64": img_b64,
            "positive_points": [
                [float(width) * 0.4, float(height) * 0.4],
                [float(width) * 0.6, float(height) * 0.6],
            ],
            "negative_points": [
                [float(width) * 0.2, float(height) * 0.2],
            ],
            "uuid": "contract",
            "image_name": image_name,
        }
        results["POST /sam_preload"] = _safe(
            "sam_preload", lambda: _post(base + "/sam_preload", {"image_base64": img_b64, "image_name": image_name})
        )
        results["POST /sam_bbox"] = _safe("sam_bbox", lambda: _post(base + "/sam_bbox", bbox_payload))
        results["POST /sam_point"] = _safe("sam_point", lambda: _post(base + "/sam_point", point_payload))
        results["POST /sam_bbox_auto"] = _safe(
            "sam_bbox_auto", lambda: _post(base + "/sam_bbox_auto", bbox_payload)
        )
        results["POST /sam_point_auto"] = _safe(
            "sam_point_auto", lambda: _post(base + "/sam_point_auto", point_payload)
        )
        results["POST /sam_point_multi"] = _safe(
            "sam_point_multi", lambda: _post(base + "/sam_point_multi", multi_payload)
        )
        results["POST /sam_point_multi_auto"] = _safe(
            "sam_point_multi_auto", lambda: _post(base + "/sam_point_multi_auto", multi_payload)
        )
        results["POST /sam_activate_slot"] = _safe(
            "sam_activate_slot",
            lambda: _post(base + "/sam_activate_slot", {"image_name": image_name}),
        )
    except Exception as exc:
        results["POST /sam_bbox"] = {"ok": False, "error": str(exc)}

    # Report
    text_label_key = "GET /datasets/{id}/text_labels/{image}"
    if text_label_key in results and not results[text_label_key].get("ok"):
        err = results[text_label_key].get("error") or ""
        if "caption_not_found" in err or "HTTP 404" in err:
            results[text_label_key] = {"ok": True, "result": {"note": "caption_not_found"}}

    activate_key = "POST /sam_activate_slot"
    if activate_key in results and not results[activate_key].get("ok"):
        err = results[activate_key].get("error") or ""
        if "slot_not_found" in err or "HTTP 404" in err:
            results[activate_key] = {"ok": True, "result": {"note": "slot_not_found"}}

    failures = {k: v for k, v in results.items() if not v.get("ok")}
    summary = {
        "base": base,
        "tested": len(results),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
