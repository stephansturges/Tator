#!/usr/bin/env python3
"""Submit and collect an OpenAI Batch caption + QA smoke run.

This runner intentionally uses the OpenAI Batch API rather than synchronous
Responses calls. It uploads local images as vision files, writes one
`/v1/responses` request per image to a batch JSONL file, submits the batch, and
persists enough state to poll or resume later.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
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
DEFAULT_POLL_SECONDS = 30.0


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


def multipart_upload(
    *,
    key: str,
    path: str,
    file_path: Path,
    purpose: str,
    timeout: float = 300.0,
) -> tuple[dict[str, Any], dict[str, str]]:
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            headers = relevant_headers(getattr(resp, "headers", None))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OpenAIRequestError(
            operation="openai_file_upload_error",
            status_code=exc.code,
            detail=detail,
            headers=relevant_headers(getattr(exc, "headers", None)),
        ) from exc
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


def _json_schema_for_qa(target_qa: int) -> dict[str, Any]:
    del target_qa
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
) -> str:
    hints = list(label_hints or [])
    if max_boxes > 0 and len(hints) > max_boxes:
        hints = hints[:max_boxes]
    return (
        "You are creating a vision training row from one drone or overhead image.\n"
        "Use the image as the source of truth. Use the annotation context only as priors.\n"
        "Use canonical glossary terms for labeled classes. Never output raw labelmap spellings.\n"
        "If a class has no glossary entry, use the natural English term.\n"
        "Return only one valid JSON object with keys caption and qa_pairs.\n"
        f"The caption must be concrete, grounded, and concise but detailed enough for training.\n"
        f"Generate exactly {target_qa} question-answer pairs in qa_pairs.\n"
        "Questions should be useful for image understanding: objects, counts, spatial relationships, visible attributes, and uncertainty when relevant.\n"
        "Answers may use grounded inference or say that something is not visible/cannot be determined when appropriate.\n"
        "Do not mention labels, boxes, prompts, coordinates, or that annotations were provided.\n\n"
        f"Image name: {case.get('stem') or case.get('name')}\n"
        f"Authoritative object counts: {json.dumps(class_counts, sort_keys=True)}\n"
        f"Glossary context: {json.dumps(glossary_context, sort_keys=True)}\n"
        f"Representative annotation boxes: {json.dumps(hints, sort_keys=True)}\n\n"
        "JSON schema:\n"
        "{\n"
        '  "caption": "final caption text",\n'
        '  "qa_pairs": [\n'
        '    {"question": "question text?", "answer": "answer text"}\n'
        "  ]\n"
        "}\n"
    )


def build_batch_line(
    *,
    case: Mapping[str, Any],
    file_id: str,
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
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
    raw_hints = _label_hints_for_case(case, dataset_root)
    canonical_hints = _canonicalize_label_hints(raw_hints, case=case, args=payload_args)
    prompt = build_prompt(
        case=case,
        label_hints=canonical_hints,
        glossary_context=caption_runner._case_glossary_context(case, payload_args),
        class_counts=_canonical_counts(case, payload_args),
        target_qa=args.qa_count,
        max_boxes=args.max_boxes,
    )
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
                "schema": _json_schema_for_qa(args.qa_count),
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


def normalize_qa_pairs(raw_pairs: Any, *, target: int) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_pairs if isinstance(raw_pairs, list) else []:
        if not isinstance(item, Mapping):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        if not question.endswith("?"):
            question = f"{question}?"
        key = re.sub(r"\s+", " ", question.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"question": question, "answer": answer})
        if len(pairs) >= target:
            break
    return pairs


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


def collect_results(*, cases: Sequence[Mapping[str, Any]], output_dir: Path, target_qa: int) -> dict[str, Any]:
    case_by_id = {case_key(case): dict(case) for case in cases}
    captions_path = output_dir / "captions.jsonl"
    results_path = output_dir / "results.jsonl"
    if captions_path.exists():
        captions_path.unlink()
    if results_path.exists():
        results_path.unlink()
    totals = Counter()
    output_rows = read_jsonl(output_dir / "batch_output.jsonl")
    for row in output_rows:
        cid = str(row.get("custom_id") or "")
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        error = row.get("error") if isinstance(row.get("error"), Mapping) else None
        body = response.get("body") if isinstance(response.get("body"), Mapping) else {}
        status_code = response.get("status_code")
        text = extract_response_text(body)
        parsed = parse_caption_payload(text)
        qa_pairs = normalize_qa_pairs(parsed.get("qa_pairs"), target=target_qa)
        caption = str(parsed.get("caption") or "").strip()
        final_status = "ok" if status_code == 200 and caption and len(qa_pairs) == target_qa and not error else "failed"
        totals[final_status] += 1
        result = {
            "case_id": cid,
            "image_name": Path(str((case_by_id.get(cid) or {}).get("image_path") or "")).name,
            "status_code": status_code,
            "final_status": final_status,
            "caption_chars": len(caption),
            "generated_qa_pair_count": len(qa_pairs),
            "generated_qa_target_pair_count": target_qa,
            "error": error,
            "response_id": body.get("id"),
            "usage": body.get("usage") or {},
        }
        append_jsonl(results_path, result)
        if caption:
            append_jsonl(
                captions_path,
                {
                    "case_id": cid,
                    "image_name": result["image_name"],
                    "image_path": (case_by_id.get(cid) or {}).get("image_path"),
                    "caption": caption,
                    "generated_qa_pairs": qa_pairs,
                    "generated_qa_pair_count": len(qa_pairs),
                    "generated_qa_target_pair_count": target_qa,
                    "raw_output_text": text,
                    "usage": body.get("usage") or {},
                },
            )
    error_rows = read_jsonl(output_dir / "batch_error.jsonl")
    for row in error_rows:
        cid = str(row.get("custom_id") or "")
        totals["failed"] += 1
        append_jsonl(
            results_path,
            {
                "case_id": cid,
                "final_status": "failed",
                "error": row.get("error"),
                "generated_qa_pair_count": 0,
                "generated_qa_target_pair_count": target_qa,
            },
        )
    summary = {
        "updated_at": utc_now(),
        "total_cases": len(cases),
        "output_rows": len(output_rows),
        "error_rows": len(error_rows),
        "totals": dict(totals),
        "failed_cases": totals.get("failed", 0),
        "caption_rows": sum(1 for _ in read_jsonl(captions_path)),
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
    parser.add_argument("--max-output-tokens", type=int, default=3200)
    parser.add_argument("--upload-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.qa_count = max(0, min(int(args.qa_count or 0), 20))
    args.max_boxes = max(0, int(args.max_boxes or 0))
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
        summary = collect_results(cases=case_rows, output_dir=args.output_dir, target_qa=args.qa_count)
        print(json.dumps({"event": "batch_collected", "summary": summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenAIRequestError as exc:
        raise SystemExit(2) from exc
