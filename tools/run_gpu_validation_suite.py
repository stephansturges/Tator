#!/usr/bin/env python3
"""Comprehensive GPU API validation suite with run-scoped artifact cleanup.

Design goals:
- Validate all GPU-capable endpoint groups in one run.
- Keep all generated local artifacts namespaced by run_id.
- Clean up only run-scoped artifacts (never touch original data).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


TERMINAL_JOB_STATES = {"completed", "succeeded", "failed", "cancelled", "canceled"}


@dataclass
class CheckResult:
    check_id: str
    phase: str
    method: str
    path: str
    ok: bool
    status_code: Optional[int] = None
    classification: Optional[str] = None
    detail: Optional[str] = None
    duration_s: Optional[float] = None
    rerun_of: Optional[str] = None
    response_excerpt: Optional[str] = None
    payload_excerpt: Optional[str] = None


@dataclass
class RunContext:
    run_id: str
    repo_root: Path
    base_url: str
    artifact_root: Path
    upload_ns_root: Path
    sample_image_path: Path
    sample_image_b64: str
    labelmap: List[str]
    glossary: str
    dataset_id: str
    dataset_classes: List[Dict[str, Any]]
    classifier_path: Optional[str]
    clip_active_payload: Dict[str, Any]
    yolo_active_payload: Dict[str, Any]
    rfdetr_active_payload: Dict[str, Any]
    first_recipe: Optional[Dict[str, Any]] = None
    created_dataset_ids: List[str] = field(default_factory=list)
    started_jobs: Dict[str, List[str]] = field(default_factory=dict)
    cleanup_manifest: Dict[str, Any] = field(default_factory=dict)


class GpuValidationSuite:
    def __init__(self, *, repo_root: Path, base_url: str, timeout_s: int, run_id: Optional[str], cleanup: bool):
        self.repo_root = repo_root.resolve()
        self.base_url = base_url.rstrip("/")
        self.timeout_s = max(5, int(timeout_s))
        self.cleanup_enabled = bool(cleanup)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        stamp = run_id or dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"gpu_validation_{stamp}"
        self.artifact_root = self.repo_root / "tmp" / self.run_id
        self.upload_ns_root = self.repo_root / "uploads" / "gpu_validation" / stamp
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.upload_ns_root.mkdir(parents=True, exist_ok=True)

        self.events_path = self.artifact_root / "events.jsonl"
        self.results_path = self.artifact_root / "results.json"
        self.cleanup_manifest_path = self.artifact_root / "cleanup_manifest.json"
        self.summary_md_path = self.artifact_root / "summary.md"

        self.results: List[CheckResult] = []
        self.failures: List[CheckResult] = []
        self.ctx: Optional[RunContext] = None

    # ---------- low-level helpers ----------
    def _now(self) -> str:
        return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _emit_event(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", self._now())
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _json_excerpt(self, payload: Any, *, max_len: int = 600) -> str:
        try:
            text = json.dumps(payload, ensure_ascii=True)
        except Exception:
            text = str(payload)
        if len(text) > max_len:
            return text[: max_len - 3] + "..."
        return text

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Dict[str, Any]] = None,
        form_payload: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[Optional[int], Any, float, Optional[str]]:
        url = f"{self.base_url}{path}"
        started = time.time()
        try:
            resp = self.session.request(
                method=method,
                url=url,
                json=json_payload,
                data=form_payload,
                files=files,
                timeout=timeout or self.timeout_s,
            )
            elapsed = time.time() - started
            body: Any
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return resp.status_code, body, elapsed, None
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - started
            return None, None, elapsed, f"request_failed:{exc}"

    def _classify_failure(self, *, status_code: Optional[int], detail: str) -> str:
        text = (detail or "").lower()
        if "cuda" in text or "out of memory" in text or "device-side" in text:
            return "CONTENTION_RESOURCE"
        if status_code is None:
            return "TRANSIENT_ENV"
        if 500 <= status_code < 600:
            if "timeout" in text or "tempor" in text or "resource" in text:
                return "CONTENTION_RESOURCE"
            return "DEFECT_LOGIC"
        if status_code in {400, 404, 409, 422, 428}:
            # For smoke validation this usually indicates payload/precondition mismatch.
            return "DEFECT_INTERFACE"
        return "TRANSIENT_ENV"

    def _validate_no_nan_inf(self, payload: Any) -> bool:
        def walk(v: Any) -> bool:
            if isinstance(v, dict):
                return all(walk(x) for x in v.values())
            if isinstance(v, list):
                return all(walk(x) for x in v)
            if isinstance(v, float):
                # Keep it explicit; avoid numpy dependency here.
                if v != v:
                    return False
                if v in {float("inf"), float("-inf")}:
                    return False
            return True

        return walk(payload)

    def _record(
        self,
        *,
        check_id: str,
        phase: str,
        method: str,
        path: str,
        status_code: Optional[int],
        body: Any,
        duration_s: float,
        ok: bool,
        detail: Optional[str],
        classification: Optional[str] = None,
        payload_excerpt: Optional[str] = None,
        rerun_of: Optional[str] = None,
    ) -> CheckResult:
        excerpt = self._json_excerpt(body) if body is not None else None
        result = CheckResult(
            check_id=check_id,
            phase=phase,
            method=method,
            path=path,
            ok=ok,
            status_code=status_code,
            classification=classification,
            detail=detail,
            duration_s=duration_s,
            rerun_of=rerun_of,
            response_excerpt=excerpt,
            payload_excerpt=payload_excerpt,
        )
        self.results.append(result)
        if not ok:
            self.failures.append(result)
        self._emit_event(
            {
                "type": "check_result",
                "check_id": check_id,
                "phase": phase,
                "method": method,
                "path": path,
                "ok": ok,
                "status_code": status_code,
                "classification": classification,
                "detail": detail,
                "duration_s": duration_s,
                "rerun_of": rerun_of,
                "response_excerpt": excerpt,
            }
        )
        return result

    # ---------- bootstrap ----------
    def _load_fixture_image(self, explicit_path: Optional[str]) -> Path:
        candidates: List[Path] = []
        if explicit_path:
            candidates.append((self.repo_root / explicit_path).resolve())
        candidates.extend(
            [
                self.repo_root / "tests" / "fixtures" / "fuzz_pack" / "images" / "img_0.png",
                self.repo_root / "tests" / "fixtures" / "fuzz_pack" / "images" / "img_1.png",
            ]
        )
        for path in candidates:
            if path.exists():
                return path
        raise RuntimeError("no_fixture_image_found")

    def _load_glossary(self) -> str:
        glossary_path = self.repo_root / "tests" / "fixtures" / "fuzz_pack" / "glossary.json"
        if glossary_path.exists():
            return glossary_path.read_text(encoding="utf-8")
        return "{}"

    def _read_labelmap(self) -> List[str]:
        status, body, _elapsed, _err = self._request("GET", "/yolo/active", timeout=15)
        if status == 200 and isinstance(body, dict):
            labelmap_path = body.get("labelmap_path")
            if isinstance(labelmap_path, str) and labelmap_path.strip():
                path = Path(labelmap_path).expanduser()
                if not path.is_absolute():
                    path = (self.repo_root / path).resolve()
                if path.exists():
                    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if lines:
                        return lines
        return ["object"]

    def _create_test_dataset_zip(self, *, image_path: Path) -> Path:
        ds_root = self.artifact_root / "dataset_src"
        for rel in [
            "train/images",
            "train/labels",
            "val/images",
            "val/labels",
        ]:
            (ds_root / rel).mkdir(parents=True, exist_ok=True)

        train_name = "sample_train.jpg"
        val_name = "sample_val.jpg"
        train_dst = ds_root / "train" / "images" / train_name
        val_dst = ds_root / "val" / "images" / val_name
        shutil.copy2(image_path, train_dst)
        shutil.copy2(image_path, val_dst)

        # YOLO labels: one centered box (class 0).
        (ds_root / "train" / "labels" / "sample_train.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        (ds_root / "val" / "labels" / "sample_val.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        (ds_root / "labelmap.txt").write_text("object\n", encoding="utf-8")
        (ds_root / "dataset.json").write_text(
            json.dumps(
                {
                    "id": self.run_id,
                    "label": self.run_id,
                    "type": "bbox",
                    "created_at": time.time(),
                    "source": "gpu_validation_suite",
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

        zip_path = self.artifact_root / "dataset_upload.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(ds_root.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, arcname=str(file_path.relative_to(ds_root)))
        return zip_path

    def _upload_test_dataset(self, zip_path: Path) -> str:
        dataset_id = f"{self.run_id}_ds"
        with zip_path.open("rb") as handle:
            status, body, elapsed, err = self._request(
                "POST",
                "/datasets/upload",
                form_payload={"dataset_id": dataset_id, "dataset_type": "bbox", "context": self.run_id},
                files={"file": (zip_path.name, handle, "application/zip")},
                timeout=max(30, self.timeout_s),
            )
        ok = bool(status == 200 and isinstance(body, dict) and body.get("id"))
        detail = err or (None if ok else f"upload_failed:{body}")
        self._record(
            check_id="SETUP-DATASET-UPLOAD",
            phase="setup",
            method="POST",
            path="/datasets/upload",
            status_code=status,
            body=body,
            duration_s=elapsed,
            ok=ok,
            detail=detail,
            classification=(None if ok else self._classify_failure(status_code=status, detail=str(detail))),
            payload_excerpt=f"dataset_id={dataset_id}",
        )
        if not ok:
            raise RuntimeError(f"dataset_upload_failed:{detail}")
        return str(body.get("id"))

    def _fetch_dataset_categories(self, dataset_id: str) -> List[Dict[str, Any]]:
        status, body, _elapsed, _err = self._request("POST", f"/sam3/datasets/{dataset_id}/convert", timeout=60)
        if status == 200 and isinstance(body, dict):
            cats = body.get("categories")
            if isinstance(cats, list) and cats:
                return [c for c in cats if isinstance(c, dict)]
        return [{"id": 1, "name": "object"}]

    def _fetch_first_classifier_path(self) -> Optional[str]:
        status, body, _elapsed, _err = self._request("GET", "/clip/classifiers")
        if status != 200:
            return None
        entries = body if isinstance(body, list) else (body.get("classifiers") if isinstance(body, dict) else [])
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if isinstance(entry, dict):
                path = entry.get("path") or entry.get("rel_path")
                if isinstance(path, str) and path:
                    return path
            elif isinstance(entry, str) and entry:
                return entry
        return None

    def _fetch_first_recipe(self) -> Optional[Dict[str, Any]]:
        status, body, _elapsed, _err = self._request("GET", "/agent_mining/recipes")
        if status == 200 and isinstance(body, list) and body:
            for item in body:
                if isinstance(item, dict):
                    return item
        return None

    def bootstrap(self, *, fixture_image: Optional[str]) -> RunContext:
        sample_path = self._load_fixture_image(fixture_image)
        sample_b64 = base64.b64encode(sample_path.read_bytes()).decode("utf-8")
        labelmap = self._read_labelmap()
        glossary = self._load_glossary()

        zip_path = self._create_test_dataset_zip(image_path=sample_path)
        dataset_id = self._upload_test_dataset(zip_path)
        dataset_classes = self._fetch_dataset_categories(dataset_id)
        classifier_path = self._fetch_first_classifier_path()
        first_recipe = self._fetch_first_recipe()

        def _must_get(path: str) -> Dict[str, Any]:
            status, body, elapsed, err = self._request("GET", path)
            ok = status == 200 and isinstance(body, dict)
            self._record(
                check_id=f"SETUP-{path.strip('/').replace('/', '-').upper()}",
                phase="setup",
                method="GET",
                path=path,
                status_code=status,
                body=body,
                duration_s=elapsed,
                ok=ok,
                detail=err if err else (None if ok else f"unexpected_response:{body}"),
                classification=(None if ok else self._classify_failure(status_code=status, detail=str(body))),
            )
            if not ok:
                raise RuntimeError(f"setup_failed:{path}")
            return body

        clip_active = _must_get("/clip/active_model")
        yolo_active = _must_get("/yolo/active")
        rfdetr_active = _must_get("/rfdetr/active")
        active_classifier = clip_active.get("classifier_path") if isinstance(clip_active, dict) else None
        if isinstance(active_classifier, str) and active_classifier.strip():
            classifier_path = active_classifier

        cleanup_manifest = {
            "run_id": self.run_id,
            "created_at": self._now(),
            "artifact_root": str(self.artifact_root),
            "upload_ns_root": str(self.upload_ns_root),
            "dataset_ids": [dataset_id],
            "job_ids": {},
            "paths": [],
        }

        self.ctx = RunContext(
            run_id=self.run_id,
            repo_root=self.repo_root,
            base_url=self.base_url,
            artifact_root=self.artifact_root,
            upload_ns_root=self.upload_ns_root,
            sample_image_path=sample_path,
            sample_image_b64=sample_b64,
            labelmap=labelmap,
            glossary=glossary,
            dataset_id=dataset_id,
            dataset_classes=dataset_classes,
            classifier_path=classifier_path,
            clip_active_payload=clip_active,
            yolo_active_payload=yolo_active,
            rfdetr_active_payload=rfdetr_active,
            first_recipe=first_recipe,
            created_dataset_ids=[dataset_id],
            started_jobs={},
            cleanup_manifest=cleanup_manifest,
        )
        self._emit_event({"type": "bootstrap_complete", "dataset_id": dataset_id, "classifier_path": classifier_path})
        return self.ctx

    # ---------- check runners ----------
    def _run_json_check(
        self,
        *,
        check_id: str,
        phase: str,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        expected_status: Sequence[int] = (200,),
        validate_numeric: bool = True,
    ) -> CheckResult:
        status, body, elapsed, err = self._request(method, path, json_payload=payload)
        payload_excerpt = self._json_excerpt(payload) if payload else None
        ok = err is None and status in set(expected_status)
        detail = err
        if ok and validate_numeric and body is not None and not self._validate_no_nan_inf(body):
            ok = False
            detail = "nan_or_inf_detected"
        if not ok and detail is None:
            detail = f"unexpected_status:{status}; body={self._json_excerpt(body)}"
        classification = None if ok else self._classify_failure(status_code=status, detail=str(detail))
        return self._record(
            check_id=check_id,
            phase=phase,
            method=method,
            path=path,
            status_code=status,
            body=body,
            duration_s=elapsed,
            ok=ok,
            detail=detail,
            classification=classification,
            payload_excerpt=payload_excerpt,
        )

    def _start_job_and_validate(
        self,
        *,
        check_id: str,
        phase: str,
        start_path: str,
        start_payload: Dict[str, Any],
        get_path_tmpl: str,
        cancel_path_tmpl: Optional[str] = None,
        should_cancel: bool = False,
        timeout_s: int = 120,
        poll_s: float = 2.0,
        job_id_key: str = "job_id",
    ) -> CheckResult:
        # Start
        status, body, elapsed, err = self._request("POST", start_path, json_payload=start_payload, timeout=max(30, self.timeout_s))
        payload_excerpt = self._json_excerpt(start_payload)
        if err or status != 200 or not isinstance(body, dict) or not body.get(job_id_key):
            detail = err or f"job_start_failed:{self._json_excerpt(body)}"
            classification = self._classify_failure(status_code=status, detail=str(detail))
            return self._record(
                check_id=check_id,
                phase=phase,
                method="POST",
                path=start_path,
                status_code=status,
                body=body,
                duration_s=elapsed,
                ok=False,
                detail=detail,
                classification=classification,
                payload_excerpt=payload_excerpt,
            )

        job_id = str(body.get(job_id_key))
        if self.ctx is not None:
            self.ctx.started_jobs.setdefault(start_path, []).append(job_id)
            self.ctx.cleanup_manifest.setdefault("job_ids", {}).setdefault(start_path, []).append(job_id)
        self._emit_event({"type": "job_started", "check_id": check_id, "start_path": start_path, "job_id": job_id})

        # Optional cancel path check.
        if should_cancel and cancel_path_tmpl:
            cancel_path = cancel_path_tmpl.format(job_id=job_id)
            c_status, c_body, c_elapsed, c_err = self._request("POST", cancel_path, json_payload={})
            c_ok = c_err is None and c_status == 200
            if not c_ok:
                detail = c_err or f"cancel_failed:{self._json_excerpt(c_body)}"
                classification = self._classify_failure(status_code=c_status, detail=str(detail))
                return self._record(
                    check_id=check_id,
                    phase=phase,
                    method="POST",
                    path=cancel_path,
                    status_code=c_status,
                    body=c_body,
                    duration_s=c_elapsed,
                    ok=False,
                    detail=detail,
                    classification=classification,
                    payload_excerpt=payload_excerpt,
                )

        # Poll state.
        deadline = time.time() + max(10, int(timeout_s))
        final_status = None
        final_body: Any = None
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            g_path = get_path_tmpl.format(job_id=job_id)
            g_status, g_body, _g_elapsed, g_err = self._request("GET", g_path, timeout=20)
            if g_err is None and g_status == 200 and isinstance(g_body, dict):
                state = str(g_body.get("status") or "").lower()
                if state:
                    final_status = state
                    final_body = g_body
                    if state in TERMINAL_JOB_STATES:
                        break
            time.sleep(max(0.25, float(poll_s)))

        ok = final_status is not None
        detail = None
        classification = None
        if not ok:
            detail = f"job_poll_timeout:job_id={job_id}"
            classification = "TRANSIENT_ENV"
        else:
            if should_cancel:
                ok = final_status in {"cancelled", "canceled", "cancelling", "failed", "completed", "succeeded"}
            else:
                ok = final_status in TERMINAL_JOB_STATES
            if not ok:
                detail = f"unexpected_job_state:{final_status}"
                classification = "DEFECT_LOGIC"

        return self._record(
            check_id=check_id,
            phase=phase,
            method="POST",
            path=start_path,
            status_code=200,
            body={"job_id": job_id, "final_status": final_status, "attempts": attempts, "job": final_body},
            duration_s=elapsed,
            ok=ok,
            detail=detail,
            classification=classification,
            payload_excerpt=payload_excerpt,
        )

    def run_control_plane(self) -> None:
        assert self.ctx is not None
        ctx = self.ctx

        # GPU-aware observability endpoints are included in strict coverage.
        self._run_json_check(check_id="OBS-001", phase="observability", method="GET", path="/system/gpu", payload=None)
        self._run_json_check(check_id="OBS-002", phase="observability", method="GET", path="/predictor_settings", payload=None)
        self._run_json_check(check_id="OBS-003", phase="observability", method="GET", path="/system/health_summary", payload=None)

        self._run_json_check(check_id="CTRL-001", phase="control", method="POST", path="/runtime/unload", payload={})
        self._run_json_check(check_id="CTRL-002", phase="control", method="POST", path="/qwen/unload", payload={})

        self._run_json_check(
            check_id="CTRL-003",
            phase="control",
            method="POST",
            path="/sam3/models/activate",
            payload={"checkpoint_path": None, "label": "Base SAM3", "enable_segmentation": True},
        )
        self._run_json_check(
            check_id="CTRL-004",
            phase="control",
            method="POST",
            path="/qwen/models/activate",
            payload={"model_id": "default"},
        )

        clip_payload = {
            "classifier_path": str(ctx.clip_active_payload.get("classifier_path") or ""),
            "labelmap_path": str(ctx.clip_active_payload.get("labelmap_path") or ""),
            "clip_model": str(ctx.clip_active_payload.get("clip_model") or ""),
        }
        self._run_json_check(
            check_id="CTRL-005",
            phase="control",
            method="POST",
            path="/clip/active_model",
            payload=clip_payload,
        )

        preload_name = f"{ctx.run_id}_preload.png"
        preload_result = self._run_json_check(
            check_id="CTRL-006",
            phase="control",
            method="POST",
            path="/sam_preload",
            payload={
                "image_base64": ctx.sample_image_b64,
                "image_name": preload_name,
                "sam_variant": "sam1",
                "slot": "current",
            },
        )

        # Use current slot name/image for activation to avoid slot routing ambiguity.
        status, body, elapsed, err = self._request("GET", "/sam_slots")
        slot_name = None
        variant = "sam1"
        if status == 200 and isinstance(body, list):
            for item in body:
                if isinstance(item, dict) and item.get("image_name"):
                    slot_name = item.get("image_name")
                    variant = str(item.get("variant") or variant)
                    break
        payload = {"image_name": slot_name or preload_name, "sam_variant": variant}
        ok = err is None and status == 200 and slot_name is not None
        self._record(
            check_id="CTRL-006A",
            phase="control",
            method="GET",
            path="/sam_slots",
            status_code=status,
            body=body,
            duration_s=elapsed,
            ok=ok,
            detail=err if err else (None if ok else "sam_slots_missing_active_image"),
            classification=(None if ok else self._classify_failure(status_code=status, detail=str(body))),
            payload_excerpt=None,
        )
        self._run_json_check(
            check_id="CTRL-007",
            phase="control",
            method="POST",
            path="/sam_activate_slot",
            payload=payload,
        )

    def run_inference(self) -> None:
        assert self.ctx is not None
        ctx = self.ctx

        common_img = {"image_base64": ctx.sample_image_b64}
        common_token = {"image_base64": ctx.sample_image_b64, "uuid": ctx.run_id}

        self._run_json_check(check_id="INF-001", phase="inference", method="POST", path="/predict_base64", payload=common_token)

        self._run_json_check(
            check_id="INF-002",
            phase="inference",
            method="POST",
            path="/yolo/predict_full",
            payload={**common_img, "conf": 0.1, "iou": 0.5, "max_det": 100},
        )
        self._run_json_check(
            check_id="INF-003",
            phase="inference",
            method="POST",
            path="/yolo/predict_windowed",
            payload={**common_img, "conf": 0.1, "iou": 0.5, "slice_size": 128, "overlap": 0.2, "merge_iou": 0.5},
        )
        self._run_json_check(
            check_id="INF-004",
            phase="inference",
            method="POST",
            path="/yolo/predict_region",
            payload={**common_img, "region": [0, 0, 64, 64], "conf": 0.1, "iou": 0.5},
        )

        self._run_json_check(
            check_id="INF-005",
            phase="inference",
            method="POST",
            path="/rfdetr/predict_full",
            payload={**common_img, "conf": 0.1, "max_det": 100},
        )
        self._run_json_check(
            check_id="INF-006",
            phase="inference",
            method="POST",
            path="/rfdetr/predict_windowed",
            payload={**common_img, "conf": 0.1, "slice_size": 128, "overlap": 0.2, "merge_iou": 0.5},
        )
        self._run_json_check(
            check_id="INF-007",
            phase="inference",
            method="POST",
            path="/rfdetr/predict_region",
            payload={**common_img, "region": [0, 0, 64, 64], "conf": 0.1},
        )

        self._run_json_check(
            check_id="INF-008",
            phase="inference",
            method="POST",
            path="/sam3/text_prompt",
            payload={
                **common_img,
                "text_prompt": "object",
                "threshold": 0.1,
                "mask_threshold": 0.1,
                "sam_variant": "sam3",
                "max_results": 10,
            },
        )
        self._run_json_check(
            check_id="INF-009",
            phase="inference",
            method="POST",
            path="/sam3/text_prompt_auto",
            payload={
                **common_img,
                "text_prompt": "object",
                "threshold": 0.1,
                "mask_threshold": 0.1,
                "sam_variant": "sam3",
                "max_results": 10,
            },
        )
        self._run_json_check(
            check_id="INF-010",
            phase="inference",
            method="POST",
            path="/sam3/visual_prompt",
            payload={
                **common_img,
                "bbox_left": 5,
                "bbox_top": 5,
                "bbox_width": 60,
                "bbox_height": 60,
                "threshold": 0.1,
                "mask_threshold": 0.1,
                "sam_variant": "sam3",
                "max_results": 10,
            },
        )

        self._run_json_check(
            check_id="INF-011",
            phase="inference",
            method="POST",
            path="/sam_point",
            payload={**common_img, "point_x": 32, "point_y": 32, "sam_variant": "sam1", "uuid": ctx.run_id},
        )
        self._run_json_check(
            check_id="INF-012",
            phase="inference",
            method="POST",
            path="/sam_point_auto",
            payload={**common_img, "point_x": 32, "point_y": 32, "sam_variant": "sam1", "uuid": ctx.run_id},
        )
        self._run_json_check(
            check_id="INF-013",
            phase="inference",
            method="POST",
            path="/sam_bbox",
            payload={
                **common_img,
                "bbox_left": 5,
                "bbox_top": 5,
                "bbox_width": 60,
                "bbox_height": 60,
                "sam_variant": "sam1",
                "uuid": ctx.run_id,
            },
        )
        self._run_json_check(
            check_id="INF-014",
            phase="inference",
            method="POST",
            path="/sam_bbox_auto",
            payload={
                **common_img,
                "bbox_left": 5,
                "bbox_top": 5,
                "bbox_width": 60,
                "bbox_height": 60,
                "sam_variant": "sam1",
                "uuid": ctx.run_id,
            },
        )
        self._run_json_check(
            check_id="INF-015",
            phase="inference",
            method="POST",
            path="/sam_point_multi",
            payload={
                **common_img,
                "positive_points": [[32, 32]],
                "negative_points": [[10, 10]],
                "sam_variant": "sam1",
                "uuid": ctx.run_id,
            },
        )
        self._run_json_check(
            check_id="INF-016",
            phase="inference",
            method="POST",
            path="/sam_point_multi_auto",
            payload={
                **common_img,
                "positive_points": [[32, 32]],
                "negative_points": [[10, 10]],
                "sam_variant": "sam1",
                "uuid": ctx.run_id,
            },
        )

        self._run_json_check(
            check_id="INF-017",
            phase="inference",
            method="POST",
            path="/qwen/infer",
            payload={
                **common_img,
                "item_list": "object",
                "prompt_type": "bbox",
                "max_results": 5,
            },
            expected_status=(200, 503),
        )
        self._run_json_check(
            check_id="INF-018",
            phase="inference",
            method="POST",
            path="/qwen/caption",
            payload={
                **common_img,
                "caption_mode": "full",
                "labelmap_glossary": ctx.glossary,
                "max_new_tokens": 64,
            },
            expected_status=(200, 503),
        )

    def run_jobs(self) -> None:
        assert self.ctx is not None
        ctx = self.ctx
        class_id = int((ctx.dataset_classes[0] or {}).get("id") or 1)
        class_name = str((ctx.dataset_classes[0] or {}).get("name") or "object")
        classifier_path = ctx.classifier_path or str(ctx.clip_active_payload.get("classifier_path") or "")

        # Prepass job path (sync endpoint).
        self._run_json_check(
            check_id="JOB-001",
            phase="jobs",
            method="POST",
            path="/qwen/prepass",
            payload={
                "dataset_id": ctx.dataset_id,
                "image_base64": ctx.sample_image_b64,
                "labelmap": ["object"],
                "labelmap_glossary": ctx.glossary,
                "enable_yolo": True,
                "enable_rfdetr": True,
                "enable_sam3_text": True,
                "enable_sam3_similarity": True,
                "prepass_only": True,
                "prepass_caption": False,
                "max_new_tokens": 128,
            },
            expected_status=(200, 503),
        )

        # Calibration
        self._start_job_and_validate(
            check_id="JOB-002",
            phase="jobs",
            start_path="/calibration/jobs",
            start_payload={
                "dataset_id": ctx.dataset_id,
                "max_images": 2,
                "seed": 42,
                "enable_yolo": True,
                "enable_rfdetr": True,
                "classifier_id": classifier_path,
                "eval_iou": 0.5,
                "label_iou": 0.5,
                "sam3_text_synonym_budget": 0,
                "similarity_exemplar_count": 1,
                "similarity_exemplar_strategy": "top",
            },
            get_path_tmpl="/calibration/jobs/{job_id}",
            cancel_path_tmpl="/calibration/jobs/{job_id}/cancel",
            should_cancel=False,
            timeout_s=300,
            poll_s=2.5,
        )

        # Agent mining lightweight run (cancel smoke).
        if classifier_path:
            self._start_job_and_validate(
                check_id="JOB-003",
                phase="jobs",
                start_path="/agent_mining/jobs",
                start_payload={
                    "dataset_id": ctx.dataset_id,
                    "classes": [class_id],
                    "eval_image_count": 1,
                    "split_seed": 42,
                    "prompt_llm_max_prompts": 0,
                    "clip_head_classifier_path": classifier_path,
                    "steps_max_steps_per_recipe": 1,
                    "steps_max_visual_seeds_per_step": 1,
                    "max_workers_per_device": 1,
                    "max_workers": 1,
                    "max_results": 20,
                    "seed_threshold": 0.01,
                    "expand_threshold": 0.1,
                    "iou_threshold": 0.5,
                },
                get_path_tmpl="/agent_mining/jobs/{job_id}",
                cancel_path_tmpl="/agent_mining/jobs/{job_id}/cancel",
                should_cancel=True,
                timeout_s=120,
                poll_s=2.0,
            )

        # Agent image apply flows.
        recipe_payload = None
        recipe_id = None
        if ctx.first_recipe and isinstance(ctx.first_recipe, dict):
            recipe_payload = ctx.first_recipe.get("recipe")
            recipe_id = ctx.first_recipe.get("id")
        if isinstance(recipe_payload, dict) and recipe_payload:
            self._run_json_check(
                check_id="JOB-004",
                phase="jobs",
                method="POST",
                path="/agent_mining/apply_image",
                payload={
                    "image_base64": ctx.sample_image_b64,
                    "recipe": recipe_payload,
                    "sam_variant": "sam3",
                    "mask_threshold": 0.1,
                    "max_results": 20,
                },
                expected_status=(200, 503),
            )
        if recipe_id:
            self._run_json_check(
                check_id="JOB-005",
                phase="jobs",
                method="POST",
                path="/agent_mining/apply_image_chain",
                payload={
                    "image_base64": ctx.sample_image_b64,
                    "sam_variant": "sam3",
                    "steps": [{"recipe_id": recipe_id, "enabled": True}],
                    "mask_threshold": 0.1,
                    "max_results": 20,
                },
                expected_status=(200, 503),
            )

        # Prompt helper jobs.
        self._start_job_and_validate(
            check_id="JOB-006",
            phase="jobs",
            start_path="/sam3/prompt_helper/jobs",
            start_payload={
                "dataset_id": ctx.dataset_id,
                "sample_per_class": 1,
                "max_synonyms": 0,
                "score_threshold": 0.05,
                "max_dets": 10,
                "iou_threshold": 0.5,
                "seed": 42,
                "use_qwen": False,
            },
            get_path_tmpl="/sam3/prompt_helper/jobs/{job_id}",
            should_cancel=False,
            timeout_s=180,
            poll_s=2.0,
        )

        self._start_job_and_validate(
            check_id="JOB-007",
            phase="jobs",
            start_path="/sam3/prompt_helper/search",
            start_payload={
                "dataset_id": ctx.dataset_id,
                "sample_per_class": 1,
                "negatives_per_class": 1,
                "score_threshold": 0.05,
                "max_dets": 10,
                "iou_threshold": 0.5,
                "seed": 42,
                "precision_floor": 0.5,
                "prompts_by_class": {str(class_id): [class_name]},
                "class_id": class_id,
            },
            get_path_tmpl="/sam3/prompt_helper/jobs/{job_id}",
            should_cancel=False,
            timeout_s=180,
            poll_s=2.0,
        )

        self._start_job_and_validate(
            check_id="JOB-008",
            phase="jobs",
            start_path="/sam3/prompt_helper/recipe",
            start_payload={
                "dataset_id": ctx.dataset_id,
                "class_id": class_id,
                "prompts": [{"prompt": class_name}],
                "sample_size": 1,
                "negatives": 1,
                "max_dets": 10,
                "iou_threshold": 0.5,
                "seed": 42,
                "score_threshold": 0.05,
                "threshold_candidates": [0.05, 0.1],
            },
            get_path_tmpl="/sam3/prompt_helper/jobs/{job_id}",
            should_cancel=False,
            timeout_s=180,
            poll_s=2.0,
        )

        # Segmentation build.
        seg_job = self._start_job_and_validate(
            check_id="JOB-009",
            phase="jobs",
            start_path="/segmentation/build/jobs",
            start_payload={
                "source_dataset_id": ctx.dataset_id,
                "output_name": f"{ctx.run_id}_seg",
                "sam_variant": "sam3",
                "output_format": "yolo-seg",
                "mask_threshold": 0.1,
                "score_threshold": 0.0,
                "max_results": 1,
                "min_size": 0,
                "simplify_epsilon": 1.0,
            },
            get_path_tmpl="/segmentation/build/jobs/{job_id}",
            should_cancel=False,
            timeout_s=300,
            poll_s=2.0,
        )
        try:
            seg_body = json.loads(seg_job.response_excerpt or "{}")
            job_payload = seg_body.get("job") if isinstance(seg_body, dict) else None
            result_payload = (job_payload or {}).get("result") if isinstance(job_payload, dict) else None
            out_dataset = (result_payload or {}).get("output_dataset_id") if isinstance(result_payload, dict) else None
            if isinstance(out_dataset, str) and out_dataset:
                ctx.created_dataset_ids.append(out_dataset)
                ctx.cleanup_manifest.setdefault("dataset_ids", []).append(out_dataset)
        except Exception:
            pass

        # Training endpoints
        run_tag = ctx.run_id
        yolo_payload = {
            "dataset_id": ctx.dataset_id,
            "run_name": f"{run_tag}_yolo",
            "task": "detect",
            "epochs": 1,
            "img_size": 128,
            "batch": 1,
            "workers": 0,
            "devices": [0],
            "accept_tos": True,
        }
        yolo_train_result = self._start_job_and_validate(
            check_id="JOB-010",
            phase="jobs",
            start_path="/yolo/train/jobs",
            start_payload=yolo_payload,
            get_path_tmpl="/yolo/train/jobs/{job_id}",
            cancel_path_tmpl="/yolo/train/jobs/{job_id}/cancel",
            should_cancel=True,
            timeout_s=120,
            poll_s=2.0,
        )

        # Head graft requires base run id.
        base_run_id = str(ctx.yolo_active_payload.get("run_id") or "")
        if base_run_id:
            self._start_job_and_validate(
                check_id="JOB-011",
                phase="jobs",
                start_path="/yolo/head_graft/jobs",
                start_payload={
                    "base_run_id": base_run_id,
                    "dataset_id": ctx.dataset_id,
                    "run_name": f"{run_tag}_head_graft",
                    "epochs": 1,
                    "img_size": 128,
                    "batch": 1,
                    "workers": 0,
                    "devices": [0],
                    "accept_tos": True,
                },
                get_path_tmpl="/yolo/head_graft/jobs/{job_id}",
                cancel_path_tmpl="/yolo/head_graft/jobs/{job_id}/cancel",
                should_cancel=True,
                timeout_s=120,
                poll_s=2.0,
            )

        self._start_job_and_validate(
            check_id="JOB-012",
            phase="jobs",
            start_path="/rfdetr/train/jobs",
            start_payload={
                "dataset_id": ctx.dataset_id,
                "run_name": f"{run_tag}_rfdetr",
                "task": "detect",
                "epochs": 1,
                "batch": 1,
                "workers": 0,
                "devices": [0],
                "accept_tos": True,
            },
            get_path_tmpl="/rfdetr/train/jobs/{job_id}",
            cancel_path_tmpl="/rfdetr/train/jobs/{job_id}/cancel",
            should_cancel=True,
            timeout_s=120,
            poll_s=2.0,
        )

        self._start_job_and_validate(
            check_id="JOB-013",
            phase="jobs",
            start_path="/sam3/train/jobs",
            start_payload={
                "dataset_id": ctx.dataset_id,
                "run_name": f"{run_tag}_sam3",
                "max_epochs": 1,
                "num_train_workers": 0,
                "num_val_workers": 0,
                "train_limit": 1,
                "val_limit": 1,
            },
            get_path_tmpl="/sam3/train/jobs/{job_id}",
            cancel_path_tmpl="/sam3/train/jobs/{job_id}/cancel",
            should_cancel=True,
            timeout_s=120,
            poll_s=2.0,
        )

        # qwen train needs qwen split; use built-in qwen_dataset.
        self._start_job_and_validate(
            check_id="JOB-014",
            phase="jobs",
            start_path="/qwen/train/jobs",
            start_payload={
                "dataset_id": "qwen_dataset",
                "run_name": f"{run_tag}_qwen",
                "max_epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "train_limit": 1,
                "val_limit": 1,
                "devices": "0",
            },
            get_path_tmpl="/qwen/train/jobs/{job_id}",
            cancel_path_tmpl="/qwen/train/jobs/{job_id}/cancel",
            should_cancel=True,
            timeout_s=120,
            poll_s=2.0,
        )

        # clip train uses native tiny dataset paths.
        train_images = str(self.repo_root / "uploads" / "datasets" / ctx.dataset_id / "train" / "images")
        train_labels = str(self.repo_root / "uploads" / "datasets" / ctx.dataset_id / "train" / "labels")
        labelmap_path = str(self.repo_root / "uploads" / "datasets" / ctx.dataset_id / "labelmap.txt")
        clip_model_filename = f"{ctx.run_id}_clip.pkl"
        clip_labelmap_filename = f"{ctx.run_id}_clip_labels.pkl"

        status, body, elapsed, err = self._request(
            "POST",
            "/clip/train",
            form_payload={
                "encoder_type": "clip",
                "clip_model_name": "ViT-B/32",
                "images_path_native": train_images,
                "labels_path_native": train_labels,
                "labelmap_path_native": labelmap_path,
                "model_filename": clip_model_filename,
                "labelmap_filename": clip_labelmap_filename,
                "batch_size": "1",
                "max_iter": "10",
                "min_per_class": "1",
                "test_size": "0.5",
                "random_seed": "42",
                "output_dir": str(ctx.upload_ns_root),
            },
            timeout=max(30, self.timeout_s),
        )
        clip_ok = bool(err is None and status == 200 and isinstance(body, dict) and body.get("job_id"))
        clip_detail = err or (None if clip_ok else f"clip_train_start_failed:{self._json_excerpt(body)}")
        clip_classification = None if clip_ok else self._classify_failure(status_code=status, detail=str(clip_detail))
        clip_result = self._record(
            check_id="JOB-015",
            phase="jobs",
            method="POST",
            path="/clip/train",
            status_code=status,
            body=body,
            duration_s=elapsed,
            ok=clip_ok,
            detail=clip_detail,
            classification=clip_classification,
            payload_excerpt=f"run_tag={ctx.run_id}",
        )
        if clip_ok:
            clip_job_id = str(body.get("job_id"))
            ctx.started_jobs.setdefault("/clip/train", []).append(clip_job_id)
            ctx.cleanup_manifest.setdefault("job_ids", {}).setdefault("/clip/train", []).append(clip_job_id)
            # Cancel quickly.
            self._run_json_check(
                check_id="JOB-015A",
                phase="jobs",
                method="POST",
                path=f"/clip/train/{clip_job_id}/cancel",
                payload={},
            )

        # Save expected clip artifacts for cleanup.
        ctx.cleanup_manifest.setdefault("paths", []).extend(
            [
                str(self.repo_root / "uploads" / "classifiers" / clip_model_filename),
                str(self.repo_root / "uploads" / "labelmaps" / clip_labelmap_filename),
                str(self.repo_root / "uploads" / "classifiers" / f"{Path(clip_model_filename).stem}.meta.pkl"),
            ]
        )

    # ---------- rerun policy ----------
    def _rerun_contention_failures(self) -> None:
        reruns: List[CheckResult] = [f for f in self.failures if f.classification == "CONTENTION_RESOURCE"]
        for failed in reruns:
            # Only rerun simple JSON checks (not job-start or setup) once.
            if failed.method not in {"GET", "POST"}:
                continue
            if failed.path.startswith("/clip/train") or failed.path.startswith("/datasets/upload"):
                continue
            status, body, elapsed, err = self._request(failed.method, failed.path, timeout=self.timeout_s)
            ok = err is None and status == 200 and self._validate_no_nan_inf(body)
            detail = err or (None if ok else f"rerun_failed:{self._json_excerpt(body)}")
            classification = None if ok else self._classify_failure(status_code=status, detail=str(detail))
            self._record(
                check_id=f"{failed.check_id}-RERUN",
                phase=f"{failed.phase}_rerun",
                method=failed.method,
                path=failed.path,
                status_code=status,
                body=body,
                duration_s=elapsed,
                ok=ok,
                detail=detail,
                classification=classification,
                rerun_of=failed.check_id,
            )

    # ---------- cleanup ----------
    def _safe_remove_path(self, path: Path, removed: List[str], skipped: List[str]) -> None:
        try:
            rp = path.resolve()
        except Exception:
            skipped.append(str(path))
            return
        allow_roots = [self.repo_root / "tmp", self.repo_root / "uploads"]
        if not any(str(rp).startswith(str(root.resolve())) for root in allow_roots):
            skipped.append(str(rp))
            return
        allowed_specific = {
            str(self.artifact_root.resolve()),
            str(self.upload_ns_root.resolve()),
        }
        if self.run_id not in str(rp) and str(rp) not in allowed_specific:
            skipped.append(str(rp))
            return
        if not rp.exists():
            return
        try:
            if rp.is_dir():
                shutil.rmtree(rp, ignore_errors=True)
            else:
                rp.unlink(missing_ok=True)
            removed.append(str(rp))
        except Exception:
            skipped.append(str(rp))

    def cleanup(self) -> Dict[str, Any]:
        assert self.ctx is not None
        ctx = self.ctx
        removed: List[str] = []
        skipped: List[str] = []

        # 1) Delete run-created datasets (API-level, safest for registry).
        for dataset_id in list(dict.fromkeys(ctx.created_dataset_ids)):
            status, body, _elapsed, err = self._request("DELETE", f"/datasets/{dataset_id}")
            ok = err is None and status == 200
            self._emit_event(
                {
                    "type": "cleanup_dataset",
                    "dataset_id": dataset_id,
                    "ok": ok,
                    "status_code": status,
                    "error": err,
                    "body": self._json_excerpt(body),
                }
            )
        # Defensive cleanup: remove any registry dataset whose id contains this run_id.
        status, body, _elapsed, _err = self._request("GET", "/datasets")
        if status == 200 and isinstance(body, list):
            for entry in body:
                if not isinstance(entry, dict):
                    continue
                dataset_id = str(entry.get("id") or "")
                if self.run_id in dataset_id:
                    self._request("DELETE", f"/datasets/{dataset_id}")

        # 2) Delete run-scoped run artifacts by name prefix where APIs exist.
        # YOLO runs
        status, body, _elapsed, _err = self._request("GET", "/yolo/runs")
        if status == 200 and isinstance(body, list):
            for entry in body:
                if not isinstance(entry, dict):
                    continue
                run_name = str(entry.get("run_name") or "")
                run_id = str(entry.get("run_id") or "")
                if self.run_id in run_name and run_id:
                    self._request("DELETE", f"/yolo/runs/{run_id}")

        # RF-DETR runs
        status, body, _elapsed, _err = self._request("GET", "/rfdetr/runs")
        if status == 200 and isinstance(body, list):
            for entry in body:
                if not isinstance(entry, dict):
                    continue
                run_name = str(entry.get("run_name") or "")
                run_id = str(entry.get("run_id") or "")
                if self.run_id in run_name and run_id:
                    self._request("DELETE", f"/rfdetr/runs/{run_id}")

        # SAM3 storage runs
        status, body, _elapsed, _err = self._request("GET", "/sam3/storage/runs")
        if status == 200 and isinstance(body, list):
            for entry in body:
                if not isinstance(entry, dict):
                    continue
                text = self._json_excerpt(entry)
                run_id = str(entry.get("id") or entry.get("run_id") or "")
                if self.run_id in text and run_id:
                    self._request("DELETE", f"/sam3/storage/runs/{run_id}")

        # 3) Remove run-tagged files/dirs directly under uploads/tmp for tools that lack delete APIs.
        search_roots = [
            self.repo_root / "uploads" / "qwen_runs" / "runs",
            self.repo_root / "uploads" / "calibration_jobs",
            self.repo_root / "uploads" / "calibration_cache" / "prepass",
            self.repo_root / "uploads" / "agent_mining",
            self.repo_root / "uploads" / "sam3_runs" / "datasets",
            self.repo_root / "uploads" / "classifiers",
            self.repo_root / "uploads" / "labelmaps",
            self.repo_root / "tmp",
            self.repo_root / "uploads" / "gpu_validation",
        ]
        for root in search_roots:
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.resolve() == self.artifact_root.resolve():
                    continue
                if self.run_id in child.name:
                    self._safe_remove_path(child, removed, skipped)

        # Explicit paths recorded in manifest.
        for path_str in ctx.cleanup_manifest.get("paths", []):
            self._safe_remove_path(Path(path_str), removed, skipped)

        # Keep local report artifacts in tmp for traceability; remove upload namespace.
        self._safe_remove_path(self.upload_ns_root, removed, skipped)

        cleanup_summary = {
            "run_id": self.run_id,
            "removed": sorted(set(removed)),
            "skipped": sorted(set(skipped)),
            "cleanup_enabled": self.cleanup_enabled,
            "finished_at": self._now(),
        }
        self.cleanup_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.cleanup_manifest_path.write_text(
            json.dumps(cleanup_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        self._emit_event({"type": "cleanup_complete", **cleanup_summary})
        return cleanup_summary

    # ---------- reports ----------
    def _results_to_dict(self) -> Dict[str, Any]:
        by_phase: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            bucket = by_phase.setdefault(r.phase, {"total": 0, "passed": 0, "failed": 0})
            bucket["total"] += 1
            if r.ok:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
        failures = [
            {
                "check_id": r.check_id,
                "phase": r.phase,
                "method": r.method,
                "path": r.path,
                "status_code": r.status_code,
                "classification": r.classification,
                "detail": r.detail,
                "rerun_of": r.rerun_of,
            }
            for r in self.results
            if not r.ok
        ]
        return {
            "run_id": self.run_id,
            "base_url": self.base_url,
            "generated_at": self._now(),
            "totals": {
                "checks": len(self.results),
                "passed": sum(1 for r in self.results if r.ok),
                "failed": sum(1 for r in self.results if not r.ok),
            },
            "by_phase": by_phase,
            "results": [r.__dict__ for r in self.results],
            "failures": failures,
            "artifacts": {
                "events_jsonl": str(self.events_path),
                "results_json": str(self.results_path),
                "cleanup_manifest_json": str(self.cleanup_manifest_path),
            },
        }

    def _write_reports(self, cleanup_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = self._results_to_dict()
        if cleanup_summary is not None:
            data["cleanup"] = cleanup_summary
        self.results_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

        lines = []
        lines.append(f"# GPU Validation Suite — {self.run_id}")
        lines.append("")
        lines.append(f"- Base URL: `{self.base_url}`")
        lines.append(f"- Generated (UTC): `{data['generated_at']}`")
        lines.append(f"- Total checks: **{data['totals']['checks']}**")
        lines.append(f"- Passed: **{data['totals']['passed']}**")
        lines.append(f"- Failed: **{data['totals']['failed']}**")
        lines.append("")
        lines.append("## By Phase")
        for phase, stats in sorted(data["by_phase"].items()):
            lines.append(f"- `{phase}`: {stats['passed']}/{stats['total']} passed")
        lines.append("")
        lines.append("## Failures")
        if not data["failures"]:
            lines.append("- none")
        else:
            for item in data["failures"]:
                lines.append(
                    "- "
                    f"`{item['check_id']}` {item['method']} `{item['path']}` "
                    f"status={item['status_code']} class={item['classification']} detail={item['detail']}"
                )
        if cleanup_summary is not None:
            lines.append("")
            lines.append("## Cleanup")
            lines.append(f"- Removed paths: {len(cleanup_summary.get('removed') or [])}")
            lines.append(f"- Skipped paths: {len(cleanup_summary.get('skipped') or [])}")
        self.summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return data

    # ---------- public entry ----------
    def run(self, *, fixture_image: Optional[str]) -> int:
        start = time.time()
        cleanup_summary = None
        exit_code = 0
        try:
            self.bootstrap(fixture_image=fixture_image)
            self.run_control_plane()
            self.run_inference()
            self.run_jobs()
            self._rerun_contention_failures()
        except Exception as exc:  # noqa: BLE001
            exit_code = 2
            self._emit_event(
                {
                    "type": "fatal_error",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                }
            )
        finally:
            if self.cleanup_enabled:
                try:
                    cleanup_summary = self.cleanup()
                except Exception as exc:  # noqa: BLE001
                    self._emit_event(
                        {
                            "type": "cleanup_error",
                            "error": str(exc),
                            "trace": traceback.format_exc(),
                        }
                    )
                    if exit_code == 0:
                        exit_code = 3
            data = self._write_reports(cleanup_summary)
            data["duration_s"] = round(time.time() - start, 3)
            self.results_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
            print(json.dumps({
                "run_id": self.run_id,
                "exit_code": exit_code,
                "totals": data.get("totals"),
                "results_json": str(self.results_path),
                "summary_md": str(self.summary_md_path),
                "cleanup_manifest_json": str(self.cleanup_manifest_path),
            }, indent=2))
        return exit_code


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive GPU API validation suite")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds")
    parser.add_argument("--run-id", default=None, help="Optional run id suffix (without gpu_validation_ prefix)")
    parser.add_argument(
        "--fixture-image",
        default=None,
        help="Optional repo-relative fixture image path (defaults to tests/fixtures/fuzz_pack images)",
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Keep run artifacts for manual inspection")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    suite = GpuValidationSuite(
        repo_root=repo_root,
        base_url=args.base_url,
        timeout_s=args.timeout,
        run_id=args.run_id,
        cleanup=not args.no_cleanup,
    )
    return suite.run(fixture_image=args.fixture_image)


if __name__ == "__main__":
    raise SystemExit(main())
