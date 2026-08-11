#!/usr/bin/env python3
"""Submit and collect an OpenAI Batch caption + QA smoke run.

This runner intentionally uses the OpenAI Batch API rather than synchronous
Responses calls. It uploads local images as vision files, writes one
`/v1/responses` request per image to a batch JSONL file, submits the batch, and
persists enough state to poll or resume later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_qwen_caption_flow_benchmark as caption_runner


API_ROOT = "https://api.openai.com/v1"
FINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

_FILE_UPLOAD_THROTTLE_LOCK = threading.Lock()
_FILE_UPLOAD_NEXT_AT = 0.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default


def _wait_for_file_upload_slot() -> None:
    global _FILE_UPLOAD_NEXT_AT
    min_interval = max(0.0, _env_float("TATOR_OPENAI_FILE_UPLOAD_MIN_INTERVAL_SECONDS", 0.08))
    if min_interval <= 0:
        return
    with _FILE_UPLOAD_THROTTLE_LOCK:
        now = time.monotonic()
        wait_s = max(0.0, _FILE_UPLOAD_NEXT_AT - now)
        _FILE_UPLOAD_NEXT_AT = max(now, _FILE_UPLOAD_NEXT_AT) + min_interval
    if wait_s > 0:
        time.sleep(wait_s)


def _retry_after_seconds(headers: Mapping[str, str], attempt: int) -> float:
    raw_retry = str(headers.get("retry-after") or "").strip()
    if raw_retry:
        try:
            return max(1.0, min(float(raw_retry), 300.0))
        except (TypeError, ValueError, OverflowError):
            pass
    return min(300.0, max(1.0, 2.0 ** min(attempt, 6)))
DEFAULT_POLL_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
MAX_OUTPUT_TOKENS = 12_000
OUTPUT_TRUNCATION_MARGIN_TOKENS = 4


class OpenAIRequestError(RuntimeError):
    def __init__(self, *, operation: str, status_code: int, detail: str, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(f"{operation}:{status_code}:{detail}")
        self.operation = operation
        self.status_code = status_code
        self.detail = detail
        self.headers = dict(headers or {})

    def payload(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.detail)
        except Exception:
            parsed = self.detail
        return {
            "operation": self.operation,
            "status_code": self.status_code,
            "detail": parsed,
            "headers": self.headers,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, Mapping):
            rows.append(dict(data))
    return rows


def api_key(path_value: str) -> str:
    env_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    key = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    if not key:
        raise SystemExit("openai_api_key_not_configured")
    return key


def relevant_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = list(headers.items())
    except Exception:
        items = []
    for raw_key, raw_value in items:
        key = str(raw_key or "").strip().lower()
        if key.startswith("x-ratelimit-") or key in {
            "retry-after",
            "x-request-id",
            "openai-processing-ms",
        }:
            out[key] = str(raw_value or "")
    return out


def request_json(
    *,
    key: str,
    method: str,
    path: str,
    body: Mapping[str, Any] | None = None,
    timeout: float = 300.0,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = json.dumps(dict(body or {})).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{API_ROOT}{path}",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            headers = relevant_headers(getattr(resp, "headers", None))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OpenAIRequestError(
            operation="openai_http_error",
            status_code=exc.code,
            detail=detail,
            headers=relevant_headers(getattr(exc, "headers", None)),
        ) from exc
    data = json.loads(text) if text.strip() else {}
    return (dict(data) if isinstance(data, Mapping) else {"raw": data}), headers


def request_file_content(*, key: str, file_id: str, timeout: float = 300.0) -> str:
    req = urllib.request.Request(
        f"{API_ROOT}/files/{urllib.parse.quote(file_id)}/content",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OpenAIRequestError(
            operation="openai_file_content_error",
            status_code=exc.code,
            detail=detail,
            headers=relevant_headers(getattr(exc, "headers", None)),
        ) from exc


def delete_uploaded_file(*, key: str, file_id: str, timeout: float = 300.0) -> dict[str, Any]:
    clean_file_id = str(file_id or "").strip()
    if not clean_file_id:
        return {"file_id": "", "status": "skipped", "reason": "missing_file_id"}
    try:
        response, headers = request_json(
            key=key,
            method="DELETE",
            path=f"/files/{urllib.parse.quote(clean_file_id)}",
            timeout=timeout,
        )
        return {
            "file_id": clean_file_id,
            "status": "deleted",
            "response": response,
            "headers": headers,
        }
    except OpenAIRequestError as exc:
        payload = exc.payload()
        return {
            "file_id": clean_file_id,
            "status": "failed",
            "error": payload,
        }


def cleanup_unsubmitted_uploaded_files(
    *,
    key: str,
    output_dir: Path,
    timeout: float = 300.0,
    reason: str = "unsubmitted_batch_cleanup",
) -> dict[str, Any]:
    """Delete remote files from a failed setup that never got a Batch id.

    Local upload manifests are archived after a successful all-file cleanup so a
    retry cannot accidentally reuse deleted file IDs.
    """

    batch = read_json(output_dir / "batch.json")
    batch_response = batch.get("response") if isinstance(batch.get("response"), Mapping) else {}
    if str(batch_response.get("id") or "").strip():
        report = {
            "status": "skipped",
            "reason": "batch_already_submitted",
            "file_count": 0,
            "deleted": 0,
            "failed": 0,
            "created_at": utc_now(),
        }
        atomic_write_json(output_dir / "orphan_file_cleanup_report.json", report)
        return report

    file_ids: list[str] = []
    for row in read_jsonl(output_dir / "image_files.jsonl"):
        file_id = str(row.get("file_id") or "").strip()
        if file_id:
            file_ids.append(file_id)
    batch_input_file = read_json(output_dir / "batch_input_file.json")
    batch_file_response = (
        batch_input_file.get("response")
        if isinstance(batch_input_file.get("response"), Mapping)
        else {}
    )
    batch_input_file_id = str(batch_file_response.get("id") or "").strip()
    if batch_input_file_id:
        file_ids.append(batch_input_file_id)

    unique_ids = list(dict.fromkeys(file_ids))
    results = [
        delete_uploaded_file(key=key, file_id=file_id, timeout=timeout)
        for file_id in unique_ids
    ]
    deleted = sum(1 for item in results if item.get("status") == "deleted")
    failed = sum(1 for item in results if item.get("status") == "failed")
    report = {
        "status": "ok" if failed == 0 else "partial_failed",
        "reason": reason,
        "file_count": len(unique_ids),
        "deleted": deleted,
        "failed": failed,
        "results": results,
        "created_at": utc_now(),
    }
    if unique_ids and failed == 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived: list[str] = []
        for name in ("image_files.jsonl", "batch_input_file.json", "batch_input.jsonl"):
            path = output_dir / name
            if not path.exists():
                continue
            archive = output_dir / f"{name}.archived_after_orphan_cleanup_{stamp}.bak"
            path.replace(archive)
            archived.append(archive.name)
        report["archived_local_manifests"] = archived
    atomic_write_json(output_dir / "orphan_file_cleanup_report.json", report)
    return report


def multipart_upload(
    *,
    key: str,
    path: str,
    file_path: Path,
    purpose: str,
    timeout: float = 300.0,
) -> tuple[dict[str, Any], dict[str, str]]:
    max_retries = max(0, min(_env_int("TATOR_OPENAI_FILE_UPLOAD_MAX_RETRIES", 8), 20))
    last_error: OpenAIRequestError | None = None
    for attempt in range(max_retries + 1):
        boundary = f"----tator-openai-batch-{uuid.uuid4().hex}"
        mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        chunks: list[bytes] = []
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(b'Content-Disposition: form-data; name="purpose"\r\n\r\n')
        chunks.append(str(purpose).encode("utf-8"))
        chunks.append(b"\r\n")
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(file_path.read_bytes())
        chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(chunks)
        req = urllib.request.Request(
            f"{API_ROOT}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        _wait_for_file_upload_slot()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                headers = relevant_headers(getattr(resp, "headers", None))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            headers = relevant_headers(getattr(exc, "headers", None))
            last_error = OpenAIRequestError(
                operation="openai_file_upload_error",
                status_code=exc.code,
                detail=detail,
                headers=headers,
            )
            if exc.code != 429 or attempt >= max_retries:
                raise last_error from exc
            sleep_s = _retry_after_seconds(headers, attempt)
            print(
                json.dumps(
                    {
                        "event": "openai_file_upload_retry",
                        "path": path,
                        "file_name": file_path.name,
                        "attempt": attempt + 1,
                        "sleep_seconds": sleep_s,
                        "status_code": exc.code,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(sleep_s)
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("openai_file_upload_failed_without_error")
    data = json.loads(text)
    return (dict(data) if isinstance(data, Mapping) else {"raw": data}), headers


def case_key(case: Mapping[str, Any]) -> str:
    return caption_runner.case_key(case)


def _label_hints_for_case(case: Mapping[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    names = caption_runner.load_labelmap(dataset_root)
    image_path = Path(str(case.get("image_path") or ""))
    label_path = Path(str(case.get("label_path") or ""))
    with Image.open(image_path) as image:
        width, height = image.size
    return caption_runner.yolo_hints(label_path, width, height, names)


def _canonicalize_label_hints(
    hints: Sequence[Mapping[str, Any]],
    *,
    case: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    try:
        glossary_map = caption_runner._case_glossary_map(case, args)
    except Exception:
        glossary_map = {}
    canonical: list[dict[str, Any]] = []
    for hint in hints:
        row = dict(hint)
        label = str(row.get("label") or "").strip()
        if label:
            try:
                row["label"] = caption_runner._case_preferred_label(label, glossary_map)
            except Exception:
                row["label"] = caption_runner._natural_label(label)
        canonical.append(row)
    return canonical


def _canonical_counts(case: Mapping[str, Any], args: argparse.Namespace) -> dict[str, int]:
    try:
        counts = caption_runner._case_canonical_class_counts(case, args)
    except Exception:
        counts = {}
    if counts:
        return counts
    out: dict[str, int] = {}
    for raw_label, raw_count in dict(case.get("class_counts") or {}).items():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        label = caption_runner._natural_label(raw_label)
        if label and count > 0:
            out[label] = out.get(label, 0) + count
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _case_target_qa(case: Mapping[str, Any], default: int) -> int:
    raw = case.get("_openai_batch_target_qa")
    if raw is None:
        raw = case.get("_generated_qa_request_count")
    return max(0, _safe_int(raw, default))


def _case_total_qa_target(case: Mapping[str, Any], default: int) -> int:
    return max(0, _safe_int(case.get("_openai_batch_total_qa_target"), _case_target_qa(case, default)))


def _case_existing_caption(case: Mapping[str, Any]) -> str:
    return str(case.get("_openai_batch_existing_caption") or "").strip()


def _qa_key(question: str) -> str:
    clean = caption_runner._clean_instruction_qa_question_text(question)
    return re.sub(r"\s+", " ", str(clean or question or "").strip().lower())


def normalize_qa_pairs(
    raw_pairs: Any,
    *,
    target: int,
    existing_questions: Sequence[str] | None = None,
    answer_format: str = "natural",
) -> list[dict[str, str]]:
    if target <= 0:
        return []
    clean_answer_format = str(answer_format or "natural").strip().lower()
    if clean_answer_format not in {"natural", "json"}:
        clean_answer_format = "natural"
    pairs: list[dict[str, str]] = []
    seen: set[str] = {
        _qa_key(question)
        for question in (existing_questions or [])
        if _qa_key(str(question or ""))
    }
    for item in raw_pairs if isinstance(raw_pairs, list) else []:
        if not isinstance(item, Mapping):
            continue
        question = caption_runner._clean_instruction_qa_question_text(item.get("question"))
        raw_answer = item.get("answer")
        if isinstance(raw_answer, (dict, list)):
            answer = json.dumps(raw_answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            answer = str(raw_answer or "").strip()
        if not question or not answer:
            continue
        if clean_answer_format == "json":
            try:
                json.loads(answer)
            except Exception:
                continue
        key = _qa_key(question)
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"question": question, "answer": answer})
        if len(pairs) >= target:
            break
    return pairs


def missing_required_questions(
    qa_pairs: Sequence[Mapping[str, Any]],
    imposed_questions: Sequence[str] | None,
) -> list[str]:
    required = caption_runner._normalize_instruction_qa_imposed_questions(imposed_questions or [])
    if not required:
        return []
    answered = {
        _qa_key(str(pair.get("question") or ""))
        for pair in qa_pairs
        if isinstance(pair, Mapping) and _qa_key(str(pair.get("question") or ""))
    }
    return [question for question in required if _qa_key(question) not in answered]


def _usage_output_tokens(body: Mapping[str, Any]) -> int:
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    return _safe_int((usage or {}).get("output_tokens"), 0)


def _looks_output_truncated(body: Mapping[str, Any], *, max_output_tokens: int | None) -> bool:
    try:
        cap = int(max_output_tokens or 0)
    except (TypeError, ValueError, OverflowError):
        cap = 0
    if cap <= 0:
        return False
    output_tokens = _usage_output_tokens(body)
    if output_tokens >= max(1, cap - OUTPUT_TRUNCATION_MARGIN_TOKENS):
        return True
    status = str(body.get("status") or "").strip().lower()
    details = body.get("incomplete_details") if isinstance(body.get("incomplete_details"), Mapping) else {}
    reason = str((details or {}).get("reason") or "").strip().lower()
    return status == "incomplete" and reason in {"max_output_tokens", "output_limit"}


def _case_existing_qa_pairs(case: Mapping[str, Any]) -> list[dict[str, str]]:
    pairs = normalize_qa_pairs(
        case.get("_openai_batch_existing_qa_pairs"),
        target=max(0, _safe_int(case.get("_openai_batch_existing_qa_count"), 0) or 100),
    )
    if pairs:
        return pairs
    raw_pairs = case.get("_openai_batch_existing_qa_pairs")
    if not isinstance(raw_pairs, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_pairs:
        if not isinstance(item, Mapping):
            continue
        question = caption_runner._clean_instruction_qa_question_text(item.get("question"))
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        key = _qa_key(question)
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": question, "answer": answer})
    return out


QA_CATEGORY_LABELS: dict[str, str] = {
    "count_presence": "count/presence",
    "spatial_layout": "spatial/layout",
    "relationship": "relationship",
    "appearance_attribute": "appearance/attribute",
    "scene_context": "scene/context",
    "visibility_limit": "visibility/uncertainty",
    "open_visible_fact": "open visible fact",
}

QA_MIX_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "balanced": [
        ("count_presence", 0.20),
        ("spatial_layout", 0.20),
        ("relationship", 0.15),
        ("appearance_attribute", 0.15),
        ("scene_context", 0.15),
        ("visibility_limit", 0.10),
        ("open_visible_fact", 0.05),
    ],
    "scene": [
        ("scene_context", 0.25),
        ("spatial_layout", 0.25),
        ("relationship", 0.20),
        ("appearance_attribute", 0.10),
        ("count_presence", 0.10),
        ("visibility_limit", 0.05),
        ("open_visible_fact", 0.05),
    ],
    "object": [
        ("count_presence", 0.25),
        ("spatial_layout", 0.20),
        ("appearance_attribute", 0.20),
        ("relationship", 0.15),
        ("scene_context", 0.10),
        ("visibility_limit", 0.05),
        ("open_visible_fact", 0.05),
    ],
    "caption": [
        ("scene_context", 0.25),
        ("spatial_layout", 0.20),
        ("appearance_attribute", 0.15),
        ("relationship", 0.15),
        ("count_presence", 0.10),
        ("visibility_limit", 0.10),
        ("open_visible_fact", 0.05),
    ],
}


def _qa_category_plan(target_qa: int, qa_mix: str) -> dict[str, int]:
    target = max(0, int(target_qa or 0))
    clean_mix = str(qa_mix or "balanced").strip().lower()
    weights = QA_MIX_WEIGHTS.get(clean_mix) or QA_MIX_WEIGHTS["balanced"]
    if target <= 0:
        return {category: 0 for category, _weight in weights}
    raw_rows: list[tuple[str, int, float, int]] = []
    assigned = 0
    for index, (category, weight) in enumerate(weights):
        raw = float(weight) * target
        floor = int(raw)
        assigned += floor
        raw_rows.append((category, floor, raw - floor, index))
    remaining = target - assigned
    # Largest-remainder rounding keeps the plan proportional while making the
    # final category counts exactly match the requested QA count.
    for category, floor, _remainder, _index in sorted(raw_rows, key=lambda item: (-item[2], item[3]))[:remaining]:
        for row_index, row in enumerate(raw_rows):
            if row[0] == category:
                raw_rows[row_index] = (row[0], floor + 1, row[2], row[3])
                break
    return {category: count for category, count, _remainder, _index in raw_rows}


def _classify_qa_question_category(question: str) -> str:
    text = f" {re.sub(r'[^a-z0-9]+', ' ', str(question or '').lower())} "
    if re.search(r"\b(where|located|position|left|right|top|bottom|center|along)\b", text):
        return "spatial_layout"
    if re.search(r"\b(relative to|relationship|between|next to|beside|near|touching|connected|surrounding|separate|grouped)\b", text):
        return "relationship"
    if re.search(r"\b(color|colour|shape|size|roof|material|texture|marking|condition|appearance|look like|visible features)\b", text):
        return "appearance_attribute"
    if re.search(r"\b(how many|count|counted|number of|visible|present|can be seen)\b", text):
        return "count_presence"
    if re.search(r"\b(scene|setting|environment|area|type of|kind of|activity|layout|context)\b", text):
        return "scene_context"
    if re.search(r"\b(unclear|unknown|cannot|determine|visible enough|not visible|not shown|identity|motion|direction)\b", text):
        return "visibility_limit"
    return ""


def _deterministic_mask_category(case: Mapping[str, Any], question: str, plan: Mapping[str, int]) -> str:
    available = [category for category, count in plan.items() if int(count or 0) > 0]
    if not available:
        return ""
    seed = "|".join(
        [
            str(case.get("case_id") or case.get("image_name") or case.get("image_path") or case.get("stem") or ""),
            str(question or ""),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()
    return available[int(digest[:8], 16) % len(available)]


def build_qa_category_plan(
    *,
    case: Mapping[str, Any],
    target_qa: int,
    imposed_questions: Sequence[str] | None,
    qa_mix: str,
) -> dict[str, Any]:
    plan = _qa_category_plan(target_qa, qa_mix)
    imposed = caption_runner._normalize_instruction_qa_imposed_questions(imposed_questions or [])
    consumed: list[dict[str, str]] = []
    for question in imposed[: max(0, int(target_qa or 0))]:
        category = _classify_qa_question_category(question)
        if not category or int(plan.get(category) or 0) <= 0:
            category = _deterministic_mask_category(case, question, plan)
        if category and int(plan.get(category) or 0) > 0:
            plan[category] = int(plan.get(category) or 0) - 1
            consumed.append({"question": question, "category": category})
        else:
            consumed.append({"question": question, "category": "required_overflow"})
    return {
        "qa_mix": str(qa_mix or "balanced").strip().lower() if str(qa_mix or "").strip().lower() in QA_MIX_WEIGHTS else "balanced",
        "target_qa": max(0, int(target_qa or 0)),
        "imposed_question_count": len(imposed),
        "remaining_generated_slots": sum(int(count or 0) for count in plan.values()),
        "category_counts": {category: int(plan.get(category) or 0) for category in QA_CATEGORY_LABELS},
        "consumed_by_required_questions": consumed,
    }


def _qa_category_plan_prompt_text(plan: Mapping[str, Any]) -> str:
    target = int(plan.get("target_qa") or 0)
    if target <= 0:
        return ""
    consumed = plan.get("consumed_by_required_questions") if isinstance(plan.get("consumed_by_required_questions"), list) else []
    counts = plan.get("category_counts") if isinstance(plan.get("category_counts"), Mapping) else {}
    lines = [
        "QA diversity plan:",
        f"- Required user questions consume {min(len(consumed), target)} of {target} QA slot(s) first.",
    ]
    remaining = int(plan.get("remaining_generated_slots") or 0)
    if remaining > 0:
        lines.append("- After required questions, generate the remaining QA pairs with these category counts:")
        for category, label in QA_CATEGORY_LABELS.items():
            count = int(counts.get(category) or 0)
            if count > 0:
                lines.append(f"  - {label}: {count}")
    else:
        lines.append("- No extra generated category slots remain after required questions; answer required questions only.")
    lines.extend(
        [
            "- Do not include category names in the JSON output.",
            "- Do not exceed the listed category counts.",
            "- If a listed category is impossible to answer from this image, replace that slot with an open visible-fact question, not another count question.",
            "- Do not create more count/presence questions than the count/presence slot count plus required questions that are count/presence questions.",
            "- Do not ask two questions with the same answer pattern.",
            "- Prefer questions that require looking at the image, not just reading object counts.",
        ]
    )
    return "\n".join(lines) + "\n"


def _json_schema_for_qa(target_qa: int) -> dict[str, Any]:
    safe_target = max(0, int(target_qa or 0))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["caption", "qa_pairs"],
        "properties": {
            "caption": {"type": "string"},
            "qa_pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["question", "answer"],
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                },
                "minItems": safe_target,
                "maxItems": safe_target,
            },
        },
    }


def build_prompt(
    *,
    case: Mapping[str, Any],
    label_hints: Sequence[Mapping[str, Any]],
    glossary_context: Mapping[str, Any],
    class_counts: Mapping[str, int],
    target_qa: int,
    max_boxes: int,
    imposed_questions: Sequence[str] | None = None,
    restrict_speculative_language: bool = False,
    qa_mix: str = "balanced",
    answer_format: str = "natural",
    include_source_annotations: bool = True,
    strict_grounding: bool = True,
) -> str:
    hints = list(label_hints or [])
    if max_boxes > 0 and len(hints) > max_boxes:
        hints = hints[:max_boxes]
    context_hints = hints if bool(include_source_annotations) else []
    context_counts = dict(class_counts or {}) if bool(include_source_annotations) else {}
    existing_caption = _case_existing_caption(case)
    existing_qa_pairs = _case_existing_qa_pairs(case)
    total_target = _case_total_qa_target(case, target_qa)
    imposed = caption_runner._normalize_instruction_qa_imposed_questions(imposed_questions or [])
    clean_qa_mix = str(qa_mix or "balanced").strip().lower()
    if clean_qa_mix not in {"balanced", "scene", "object", "caption"}:
        clean_qa_mix = "balanced"
    qa_plan = build_qa_category_plan(
        case=case,
        target_qa=target_qa,
        imposed_questions=imposed,
        qa_mix=clean_qa_mix,
    )
    mix_text = {
        "balanced": "Use the computed balanced QA diversity plan below.",
        "scene": "Use the computed scene-heavy QA diversity plan below, emphasizing setting, layout, and conditions.",
        "object": "Use the computed object-heavy QA diversity plan below, emphasizing visible objects and read-only context.",
        "caption": "Use the computed caption-grounded QA diversity plan below, emphasizing grounded alternate descriptions.",
    }[clean_qa_mix]
    clean_answer_format = str(answer_format or "natural").strip().lower()
    if clean_answer_format not in {"natural", "json"}:
        clean_answer_format = "natural"
    answer_text = (
        'Answers must be JSON-encoded strings, for example "{\\"answer\\":\\"visible fact\\"}". '
        "Do not return raw JSON objects in the answer field."
        if clean_answer_format == "json"
        else "Answers must be concise natural-language facts, not JSON strings."
    )
    speculation_text = caption_runner._instruction_qa_speculation_policy_text(bool(restrict_speculative_language))
    grounding_text = (
        "Use strict grounding: each answer must be supported by the image, caption, glossary context, or read-only annotation context."
        if bool(strict_grounding)
        else "Prefer visual grounding and avoid unsupported facts."
    )
    source_text = (
        "Read-only annotation counts and representative boxes may be used as context, but never mentioned as annotations."
        if bool(include_source_annotations)
        else "Ignore source annotation boxes for QA content; use only the image, caption, and glossary terms."
    )
    imposed_text = ""
    if imposed:
        imposed_answer_policy = (
            "If a required question cannot be answered directly, omit it rather than using unavailable-information language. "
            if restrict_speculative_language
            else "If a required question cannot be answered directly, answer that it is not visible or cannot be determined instead of inventing it. "
        )
        imposed_text = (
            "Required user questions:\n"
            f"{json.dumps(imposed, ensure_ascii=False)}\n"
            "Answer these first and keep each question text exactly as provided when the answer is grounded. "
            f"{imposed_answer_policy}"
            "After required questions, add generated questions only if more rows are needed.\n"
        )
    catchup_block = ""
    if existing_caption or existing_qa_pairs:
        catchup_block = (
            "This is a catch-up request for an already-paid partial row.\n"
            "Keep the existing caption as the caption value unless the image clearly contradicts it.\n"
            f"Generate exactly {target_qa} additional non-duplicate question-answer pairs, so the merged row reaches {total_target} total QA pairs.\n"
            "Do not repeat any existing question.\n"
            f"Existing caption: {existing_caption}\n"
            f"Existing accepted QA pairs: {json.dumps(existing_qa_pairs, sort_keys=True)}\n\n"
        )
    qa_instruction = (
        "Do not generate question-answer pairs; return qa_pairs as an empty array.\n"
        if target_qa <= 0
        else f"Generate exactly {target_qa} question-answer pairs in qa_pairs.\n"
    )
    qa_example = (
        '  "qa_pairs": []\n'
        if target_qa <= 0
        else (
            '  "qa_pairs": [\n'
            '    {"question": "question text?", "answer": "answer text"}\n'
            "  ]\n"
        )
    )
    return (
        "You are creating a vision training row from one drone or overhead image.\n"
        "Use the image as the source of truth. Use the annotation context only as priors.\n"
        "Use glossary terms as semantic anchors for labeled classes, but do not force generic wording when the image supports a more specific natural subtype.\n"
        "For example, a glossary class like light vehicle may be described as a blue sedan, white van, red pickup, small tuk-tuk, or light vehicle depending on what is actually visible.\n"
        "Apply the same rule to every class: use a specific visible subtype when supported, and use the broad glossary term when the subtype is unclear.\n"
        "Never output raw labelmap spellings.\n"
        "If a class has no glossary entry, use the natural English term.\n"
        "Return only one valid JSON object with keys caption and qa_pairs.\n"
        f"The caption must be concrete, grounded, and concise but detailed enough for training.\n"
        f"{qa_instruction}"
        f"{imposed_text}"
        f"{mix_text}\n"
        f"{_qa_category_plan_prompt_text(qa_plan)}"
        f"{answer_text}\n"
        f"{speculation_text}\n"
        f"{grounding_text}\n"
        f"{source_text}\n"
        "Questions should be useful for image understanding: objects, counts, spatial relationships, visible attributes, and uncertainty when relevant.\n"
        "Do not mention labels, boxes, prompts, annotation coordinates, or that annotations were provided. "
        "Use image coordinates only when a required question explicitly asks for them.\n\n"
        f"{catchup_block}"
        f"Image name: {case.get('stem') or case.get('name')}\n"
        f"Authoritative object counts: {json.dumps(context_counts, sort_keys=True)}\n"
        f"Glossary context: {json.dumps(glossary_context, sort_keys=True)}\n"
        f"Representative annotation boxes: {json.dumps(context_hints, sort_keys=True)}\n\n"
        "JSON schema:\n"
        "{\n"
        '  "caption": "final caption text",\n'
        f"{qa_example}"
        "}\n"
    )


def _batch_payload_args(args: argparse.Namespace) -> argparse.Namespace:
    payload_args = argparse.Namespace(**vars(args))
    payload_args.model_id = "openai-batch"
    payload_args.refinement_model_id = "same"
    payload_args.fallback_model_id = "auto"
    payload_args.loop_recovery = "safe_retry_fallback"
    payload_args.use_sampling = False
    payload_args.temperature = 0.2
    payload_args.top_p = 0.8
    payload_args.top_k = 20
    payload_args.windowed_full_image_strategy = "visual"
    payload_args.window_size = 672
    payload_args.window_overlap = 0.1
    payload_args.final_sentences = 8
    payload_args.max_new_tokens = args.max_output_tokens
    payload_args.max_boxes = args.max_boxes
    payload_args.request_json = args.request_json
    payload_args.instruction_qa_imposed_questions = list(getattr(args, "instruction_qa_imposed_questions", []) or [])
    payload_args.instruction_qa_restrict_speculative_language = bool(
        getattr(args, "instruction_qa_restrict_speculative_language", False)
    )
    payload_args.include_source_annotations_in_generator_context = bool(
        getattr(args, "include_source_annotations_in_generator_context", True)
    )
    payload_args.strict_grounding = bool(getattr(args, "strict_grounding", True))
    payload_args.qa_mix = str(getattr(args, "qa_mix", "balanced") or "balanced")
    payload_args.answer_format = str(getattr(args, "answer_format", "natural") or "natural")
    return payload_args


def build_batch_prompt_context(
    *,
    case: Mapping[str, Any],
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload_args = _batch_payload_args(args)
    case_missing_required = caption_runner._normalize_instruction_qa_imposed_questions(
        case.get("_openai_batch_missing_required_questions")
        if isinstance(case.get("_openai_batch_missing_required_questions"), list)
        else []
    )
    prompt_imposed_questions = case_missing_required or payload_args.instruction_qa_imposed_questions
    raw_hints = _label_hints_for_case(case, dataset_root)
    canonical_hints = _canonicalize_label_hints(raw_hints, case=case, args=payload_args)
    target_qa = _case_target_qa(case, args.qa_count)
    glossary_context = caption_runner._case_glossary_context(case, payload_args)
    class_counts = _canonical_counts(case, payload_args)
    qa_category_plan = build_qa_category_plan(
        case=case,
        target_qa=target_qa,
        imposed_questions=prompt_imposed_questions,
        qa_mix=payload_args.qa_mix,
    )
    prompt = build_prompt(
        case=case,
        label_hints=canonical_hints,
        glossary_context=glossary_context,
        class_counts=class_counts,
        target_qa=target_qa,
        max_boxes=args.max_boxes,
        imposed_questions=prompt_imposed_questions,
        restrict_speculative_language=payload_args.instruction_qa_restrict_speculative_language,
        qa_mix=payload_args.qa_mix,
        answer_format=payload_args.answer_format,
        include_source_annotations=payload_args.include_source_annotations_in_generator_context,
        strict_grounding=payload_args.strict_grounding,
    )
    representative_hints = list(canonical_hints)
    if int(args.max_boxes or 0) > 0:
        representative_hints = representative_hints[: int(args.max_boxes)]
    if not payload_args.include_source_annotations_in_generator_context:
        representative_hints = []
    return {
        "payload_args": payload_args,
        "prompt": prompt,
        "target_qa": target_qa,
        "total_qa_target": _case_total_qa_target(case, target_qa),
        "imposed_questions": prompt_imposed_questions,
        "missing_required_questions": case_missing_required,
        "raw_label_hint_count": len(raw_hints),
        "canonical_label_hint_count": len(canonical_hints),
        "representative_box_count": len(representative_hints),
        "glossary_context": glossary_context,
        "class_counts": class_counts if payload_args.include_source_annotations_in_generator_context else {},
        "include_source_annotations": payload_args.include_source_annotations_in_generator_context,
        "strict_grounding": payload_args.strict_grounding,
        "qa_mix": payload_args.qa_mix,
        "qa_category_plan": qa_category_plan,
        "answer_format": payload_args.answer_format,
        "restrict_speculative_language": payload_args.instruction_qa_restrict_speculative_language,
    }


def build_batch_line(
    *,
    case: Mapping[str, Any],
    file_id: str,
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_context = build_batch_prompt_context(case=case, dataset_root=dataset_root, args=args)
    prompt = str(prompt_context.get("prompt") or "")
    target_qa = int(prompt_context.get("target_qa") or 0)
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        {
            "type": "input_image",
            "file_id": file_id,
            "detail": args.image_detail,
        },
    ]
    body: dict[str, Any] = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": args.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "caption_qa_row",
                "strict": True,
                "schema": _json_schema_for_qa(target_qa),
            }
        },
        "store": False,
    }
    return {
        "custom_id": case_key(case),
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def extract_response_text(body: Mapping[str, Any]) -> str:
    text = body.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts: list[str] = []
    for item in body.get("output") if isinstance(body.get("output"), list) else []:
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content") if isinstance(item.get("content"), list) else []:
            if not isinstance(content, Mapping):
                continue
            raw = content.get("text") or content.get("output_text")
            if isinstance(raw, str):
                parts.append(raw)
    return "\n".join(parts).strip()


def parse_caption_payload(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    candidates = [raw]
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1))
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, Mapping):
            return dict(data)
    return {"caption": raw, "qa_pairs": []}


def upload_images(
    *,
    key: str,
    cases: Sequence[Mapping[str, Any]],
    output_dir: Path,
    workers: int,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    image_files_path = output_dir / "image_files.jsonl"
    existing = {row.get("case_id"): row for row in read_jsonl(image_files_path)}
    lock_path = output_dir / "image_upload_lock.json"
    del lock_path

    def upload(case: Mapping[str, Any]) -> dict[str, Any]:
        cid = case_key(case)
        previous = existing.get(cid)
        if isinstance(previous, Mapping) and previous.get("file_id"):
            return dict(previous)
        image_path = Path(str(case.get("image_path") or ""))
        response, headers = multipart_upload(
            key=key,
            path="/files",
            file_path=image_path,
            purpose="vision",
            timeout=timeout,
        )
        return {
            "case_id": cid,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "file_id": response.get("id"),
            "bytes": image_path.stat().st_size,
            "uploaded_at": utc_now(),
            "response": response,
            "headers": headers,
        }

    completed: dict[str, dict[str, Any]] = {
        str(key_): dict(value)
        for key_, value in existing.items()
        if key_ and isinstance(value, Mapping) and value.get("file_id")
    }
    pending = [case for case in cases if case_key(case) not in completed]
    if not pending:
        return completed
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(upload, case) for case in pending]
        for future in as_completed(futures):
            row = future.result()
            completed[str(row["case_id"])] = row
            append_jsonl(image_files_path, row)
            print(json.dumps({"event": "image_uploaded", "case_id": row["case_id"], "file_id": row["file_id"]}, sort_keys=True), flush=True)
    return completed


def write_batch_input(
    *,
    cases: Sequence[Mapping[str, Any]],
    file_rows: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    path = output_dir / "batch_input.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            cid = case_key(case)
            file_id = str((file_rows.get(cid) or {}).get("file_id") or "")
            if not file_id:
                raise RuntimeError(f"missing_uploaded_file_id:{cid}")
            line = build_batch_line(
                case=case,
                file_id=file_id,
                dataset_root=dataset_root,
                args=args,
            )
            handle.write(json.dumps(line, sort_keys=True) + "\n")
    return path


def submit_batch(
    *,
    key: str,
    batch_input: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    existing = read_json(output_dir / "batch.json")
    existing_response = existing.get("response") if isinstance(existing.get("response"), Mapping) else {}
    if existing_response.get("id"):
        return existing
    existing_file = read_json(output_dir / "batch_input_file.json")
    file_response = existing_file.get("response") if isinstance(existing_file.get("response"), Mapping) else {}
    if not file_response.get("id"):
        file_response, file_headers = multipart_upload(
            key=key,
            path="/files",
            file_path=batch_input,
            purpose="batch",
            timeout=args.timeout,
        )
        atomic_write_json(
            output_dir / "batch_input_file.json",
            {"response": file_response, "headers": file_headers, "uploaded_at": utc_now()},
        )
    try:
        batch_response, batch_headers = request_json(
            key=key,
            method="POST",
            path="/batches",
            body={
                "input_file_id": file_response["id"],
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {
                    "kind": "caption_qa_smoke",
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "image_detail": args.image_detail,
                    "qa_count": str(args.qa_count),
                    "qa_mix": str(getattr(args, "qa_mix", "balanced") or "balanced"),
                    "answer_format": str(getattr(args, "answer_format", "natural") or "natural"),
                    "imposed_question_count": str(len(getattr(args, "instruction_qa_imposed_questions", []) or [])),
                    "restrict_speculative_language": "true"
                    if bool(getattr(args, "instruction_qa_restrict_speculative_language", False))
                    else "false",
                },
            },
            timeout=args.timeout,
        )
    except OpenAIRequestError as exc:
        payload = {
            "created_at": utc_now(),
            "input_file_id": file_response.get("id"),
            **exc.payload(),
        }
        atomic_write_json(output_dir / "batch_create_error.json", payload)
        print(json.dumps({"event": "batch_create_error", "status_code": exc.status_code, "detail": payload["detail"]}, sort_keys=True), flush=True)
        raise
    payload = {"response": batch_response, "headers": batch_headers, "created_at": utc_now()}
    atomic_write_json(output_dir / "batch.json", payload)
    return payload


def poll_batch(
    *,
    key: str,
    batch_id: str,
    output_dir: Path,
    poll_seconds: float,
    wait_seconds: float,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.time() + max(0.0, wait_seconds)
    latest: dict[str, Any] = {}
    while True:
        response, headers = request_json(
            key=key,
            method="GET",
            path=f"/batches/{urllib.parse.quote(batch_id)}",
            timeout=timeout,
        )
        latest = {"response": response, "headers": headers, "polled_at": utc_now()}
        atomic_write_json(output_dir / "batch_status.json", latest)
        append_jsonl(output_dir / "batch_status.jsonl", latest)
        print(json.dumps({"event": "batch_status", "id": batch_id, "status": response.get("status"), "counts": response.get("request_counts")}, sort_keys=True), flush=True)
        if str(response.get("status") or "") in FINAL_BATCH_STATUSES:
            return latest
        if wait_seconds <= 0 or time.time() >= deadline:
            return latest
        time.sleep(max(1.0, poll_seconds))


def download_outputs(*, key: str, batch: Mapping[str, Any], output_dir: Path, timeout: float) -> None:
    response = batch.get("response") if isinstance(batch.get("response"), Mapping) else batch
    output_file_id = str(response.get("output_file_id") or "")
    error_file_id = str(response.get("error_file_id") or "")
    if output_file_id and not (output_dir / "batch_output.jsonl").exists():
        (output_dir / "batch_output.jsonl").write_text(
            request_file_content(key=key, file_id=output_file_id, timeout=timeout),
            encoding="utf-8",
        )
    if error_file_id and not (output_dir / "batch_error.jsonl").exists():
        (output_dir / "batch_error.jsonl").write_text(
            request_file_content(key=key, file_id=error_file_id, timeout=timeout),
            encoding="utf-8",
        )


def collect_results(
    *,
    cases: Sequence[Mapping[str, Any]],
    output_dir: Path,
    target_qa: int,
    answer_format: str = "natural",
    imposed_questions: Sequence[str] | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    case_by_id = {case_key(case): dict(case) for case in cases}
    captions_path = output_dir / "captions.jsonl"
    incomplete_path = output_dir / "incomplete_captions.jsonl"
    results_path = output_dir / "results.jsonl"
    if captions_path.exists():
        captions_path.unlink()
    if incomplete_path.exists():
        incomplete_path.unlink()
    if results_path.exists():
        results_path.unlink()
    totals = Counter()
    output_rows = read_jsonl(output_dir / "batch_output.jsonl")
    observed_case_ids: set[str] = set()
    for row in output_rows:
        cid = str(row.get("custom_id") or "")
        if cid:
            observed_case_ids.add(cid)
        case = case_by_id.get(cid) or {}
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        error = row.get("error") if isinstance(row.get("error"), Mapping) else None
        body = response.get("body") if isinstance(response.get("body"), Mapping) else {}
        status_code = response.get("status_code")
        text = extract_response_text(body)
        parsed = parse_caption_payload(text)
        requested_qa = _case_target_qa(case, target_qa)
        total_target_qa = _case_total_qa_target(case, target_qa)
        existing_caption = _case_existing_caption(case)
        existing_qa_pairs = _case_existing_qa_pairs(case)
        new_qa_pairs = normalize_qa_pairs(
            parsed.get("qa_pairs"),
            target=requested_qa,
            existing_questions=[pair["question"] for pair in existing_qa_pairs],
            answer_format=answer_format,
        )
        qa_pairs = (existing_qa_pairs + new_qa_pairs)[:total_target_qa if total_target_qa > 0 else len(new_qa_pairs)]
        missing_imposed = missing_required_questions(qa_pairs, imposed_questions)
        caption = str(parsed.get("caption") or existing_caption or "").strip()
        output_truncated = _looks_output_truncated(body, max_output_tokens=max_output_tokens)
        failure_reason = ""
        if error:
            failure_reason = "batch_row_error"
        elif status_code != 200:
            failure_reason = f"http_status_{status_code}"
        elif output_truncated and (not caption or len(qa_pairs) != total_target_qa or missing_imposed):
            failure_reason = "output_truncated"
        elif not caption:
            failure_reason = "caption_missing"
        elif len(qa_pairs) != total_target_qa:
            failure_reason = "generated_qa_incomplete"
        elif missing_imposed:
            failure_reason = "required_qa_missing"
        final_status = "ok" if not failure_reason else ("incomplete_qa" if failure_reason in {"generated_qa_incomplete", "required_qa_missing"} and caption else "failed")
        totals[final_status] += 1
        result = {
            "case_id": cid,
            "image_name": Path(str(case.get("image_path") or "")).name,
            "status_code": status_code,
            "final_status": final_status,
            "failure_reason": failure_reason,
            "caption_chars": len(caption),
            "generated_qa_pair_count": len(qa_pairs),
            "generated_qa_new_pair_count": len(new_qa_pairs),
            "generated_qa_existing_pair_count": len(existing_qa_pairs),
            "generated_qa_requested_pair_count": requested_qa,
            "generated_qa_target_pair_count": total_target_qa,
            "answer_format": str(answer_format or "natural"),
            "missing_required_questions": missing_imposed,
            "output_truncated": output_truncated,
            "error": error,
            "response_id": body.get("id"),
            "usage": body.get("usage") or {},
        }
        append_jsonl(results_path, result)
        caption_row = {
            "case_id": cid,
            "image_name": result["image_name"],
            "image_path": case.get("image_path"),
            "caption": caption,
            "generated_qa_pairs": qa_pairs,
            "generated_qa_new_pairs": new_qa_pairs,
            "generated_qa_pair_count": len(qa_pairs),
            "generated_qa_new_pair_count": len(new_qa_pairs),
            "generated_qa_existing_pair_count": len(existing_qa_pairs),
            "generated_qa_requested_pair_count": requested_qa,
            "generated_qa_target_pair_count": total_target_qa,
            "answer_format": str(answer_format or "natural"),
            "missing_required_questions": missing_imposed,
            "output_truncated": output_truncated,
            "raw_output_text": text,
            "usage": body.get("usage") or {},
            "final_status": final_status,
            "failure_reason": failure_reason,
        }
        if existing_qa_pairs:
            caption_row["openai_batch_catchup"] = {
                "source_job_id": case.get("_openai_batch_catchup_source_job_id"),
                "existing_pair_count": len(existing_qa_pairs),
                "new_pair_count": len(new_qa_pairs),
                "total_target_pair_count": total_target_qa,
            }
        if final_status == "ok":
            append_jsonl(
                captions_path,
                caption_row,
            )
        elif caption and final_status == "incomplete_qa":
            append_jsonl(incomplete_path, caption_row)
    error_rows = read_jsonl(output_dir / "batch_error.jsonl")
    for row in error_rows:
        cid = str(row.get("custom_id") or "")
        if cid:
            observed_case_ids.add(cid)
        case = case_by_id.get(cid) or {}
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        body = response.get("body") if isinstance(response.get("body"), Mapping) else {}
        status_code = response.get("status_code")
        error = row.get("error") if isinstance(row.get("error"), Mapping) else body.get("error")
        image_name = (
            str(case.get("image_name") or "").strip()
            or Path(str(case.get("image_path") or "")).name
            or str(case.get("stem") or "").strip()
        )
        totals["failed"] += 1
        append_jsonl(
            results_path,
            {
                "case_id": cid,
                "image_name": image_name,
                "status_code": status_code,
                "final_status": "failed",
                "failure_reason": "batch_row_error",
                "error": error if isinstance(error, Mapping) else row.get("error"),
                "generated_qa_pair_count": 0,
                "generated_qa_target_pair_count": _case_total_qa_target(case, target_qa),
            },
        )
    missing_case_ids: list[str] = []
    for case in cases:
        cid = case_key(case)
        if not cid or cid in observed_case_ids:
            continue
        missing_case_ids.append(cid)
        totals["failed"] += 1
        image_name = (
            str(case.get("image_name") or "").strip()
            or Path(str(case.get("image_path") or "")).name
            or str(case.get("stem") or "").strip()
        )
        append_jsonl(
            results_path,
            {
                "case_id": cid,
                "image_name": image_name,
                "final_status": "failed",
                "failure_reason": "missing_batch_result",
                "generated_qa_pair_count": 0,
                "generated_qa_target_pair_count": _case_total_qa_target(case, target_qa),
            },
        )
    summary = {
        "updated_at": utc_now(),
        "total_cases": len(cases),
        "output_rows": len(output_rows),
        "error_rows": len(error_rows),
        "missing_result_rows": len(missing_case_ids),
        "missing_result_case_ids": missing_case_ids[:50],
        "missing_result_case_id_limit": 50,
        "totals": dict(totals),
        "failed_cases": totals.get("failed", 0),
        "incomplete_cases": totals.get("incomplete_qa", 0),
        "caption_rows": sum(1 for _ in read_jsonl(captions_path)),
        "incomplete_caption_rows": sum(1 for _ in read_jsonl(incomplete_path)),
        "accepted_cases": totals.get("ok", 0),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", default="openAI_API_KEY_DoNotCommit")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="high")
    parser.add_argument("--image-detail", choices=("original", "high", "low", "auto"), default="original")
    parser.add_argument("--qa-count", type=int, default=8)
    parser.add_argument("--max-boxes", type=int, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--instruction-qa-imposed-question",
        action="append",
        dest="instruction_qa_imposed_questions",
        default=[],
    )
    parser.add_argument("--restrict-speculative-qa-language", action="store_true", dest="instruction_qa_restrict_speculative_language")
    parser.add_argument("--qa-mix", choices=("balanced", "scene", "object", "caption"), default="balanced")
    parser.add_argument("--answer-format", choices=("natural", "json"), default="natural")
    parser.add_argument("--no-source-annotations-in-generator-context", action="store_false", dest="include_source_annotations_in_generator_context")
    parser.add_argument("--no-strict-grounding", action="store_false", dest="strict_grounding")
    parser.add_argument("--upload-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.set_defaults(include_source_annotations_in_generator_context=True, strict_grounding=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.qa_count = max(0, min(int(args.qa_count or 0), 20))
    args.max_boxes = max(0, int(args.max_boxes or 0))
    args.instruction_qa_imposed_questions = caption_runner._normalize_instruction_qa_imposed_questions(
        getattr(args, "instruction_qa_imposed_questions", None)
    )
    key = api_key(args.api_key_file)
    cases = json.loads(args.cases_json.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise SystemExit("--cases-json must contain a list")
    case_rows = [dict(case) for case in cases if isinstance(case, Mapping)]
    manifest = {
        "created_at": utc_now(),
        "cases_json": str(args.cases_json),
        "dataset_root": str(args.dataset_root),
        "request_json": str(args.request_json),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "image_detail": args.image_detail,
        "qa_count": args.qa_count,
        "max_boxes": args.max_boxes,
        "max_output_tokens": args.max_output_tokens,
        "instruction_qa_imposed_questions": list(args.instruction_qa_imposed_questions or []),
        "instruction_qa_restrict_speculative_language": bool(args.instruction_qa_restrict_speculative_language),
        "qa_mix": args.qa_mix,
        "answer_format": args.answer_format,
        "include_source_annotations_in_generator_context": bool(args.include_source_annotations_in_generator_context),
        "strict_grounding": bool(args.strict_grounding),
        "case_count": len(case_rows),
        "api": {
            "batch_endpoint": "/v1/batches",
            "underlying_endpoint": "/v1/responses",
            "image_input": "files_purpose_vision_file_id",
        },
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    file_rows = upload_images(
        key=key,
        cases=case_rows,
        output_dir=args.output_dir,
        workers=max(1, int(args.upload_workers or 1)),
        timeout=args.timeout,
    )
    batch_input = write_batch_input(
        cases=case_rows,
        file_rows=file_rows,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        args=args,
    )
    batch_input_size = batch_input.stat().st_size
    if batch_input_size > 200_000_000:
        raise SystemExit(f"batch_input_too_large:{batch_input_size}")
    batch = submit_batch(key=key, batch_input=batch_input, output_dir=args.output_dir, args=args)
    batch_id = str((batch.get("response") or {}).get("id") or "")
    if not batch_id:
        raise SystemExit("batch_id_missing")
    latest = poll_batch(
        key=key,
        batch_id=batch_id,
        output_dir=args.output_dir,
        poll_seconds=args.poll_seconds,
        wait_seconds=args.wait_seconds,
        timeout=args.timeout,
    )
    if str((latest.get("response") or {}).get("status") or "") == "completed":
        download_outputs(key=key, batch=latest, output_dir=args.output_dir, timeout=args.timeout)
        summary = collect_results(
            cases=case_rows,
            output_dir=args.output_dir,
            target_qa=args.qa_count,
            answer_format=args.answer_format,
            imposed_questions=args.instruction_qa_imposed_questions,
            max_output_tokens=args.max_output_tokens,
        )
        print(json.dumps({"event": "batch_collected", "summary": summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenAIRequestError as exc:
        raise SystemExit(2) from exc
