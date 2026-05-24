import json
import uuid
import urllib.error
import urllib.request
from typing import Any

from .env import api_root


def api_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    expected_statuses: tuple[int, ...] = (200,),
) -> Any:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url=f"{api_root()}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310
            status = int(getattr(resp, "status", 200))
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")

    if status not in expected_statuses:
        raise AssertionError(f"Unexpected status {status} for {method} {path}: {raw}")
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}


def api_multipart(
    method: str,
    path: str,
    *,
    fields: dict[str, str] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    expected_statuses: tuple[int, ...] = (200,),
) -> Any:
    boundary = f"----tator-ui-e2e-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def add_text(value: str) -> None:
        chunks.append(value.encode("utf-8"))

    for name, value in (fields or {}).items():
        add_text(f"--{boundary}\r\n")
        add_text(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        add_text(f"{value}\r\n")

    for name, (filename, payload, content_type) in (files or {}).items():
        add_text(f"--{boundary}\r\n")
        add_text(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        )
        add_text(f"Content-Type: {content_type or 'application/octet-stream'}\r\n\r\n")
        chunks.append(payload)
        add_text("\r\n")

    add_text(f"--{boundary}--\r\n")
    body = b"".join(chunks)
    req = urllib.request.Request(
        url=f"{api_root()}{path}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            status = int(getattr(resp, "status", 200))
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")

    if status not in expected_statuses:
        raise AssertionError(f"Unexpected status {status} for {method} {path}: {raw}")
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}


def delete_dataset_if_exists(dataset_id: str) -> None:
    if not dataset_id:
        return
    try:
        api_json("DELETE", f"/datasets/{dataset_id}", expected_statuses=(200, 404))
    except Exception:
        # Cleanup should not hide primary test failures.
        pass
