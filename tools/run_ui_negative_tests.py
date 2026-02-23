#!/usr/bin/env python3
import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError


def _request(method: str, url: str, body: dict | None = None):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read().decode("utf-8")


def _safe(name, method, url, body=None, ok_statuses=None):
    ok_statuses = set(ok_statuses or [])
    try:
        status, payload = _request(method, url, body)
        return {"ok": True, "status": status, "payload": payload[:2000]}
    except HTTPError as exc:
        body_txt = exc.read().decode("utf-8") if exc.fp else ""
        if exc.code in ok_statuses:
            return {"ok": True, "status": exc.code, "payload": body_txt[:2000]}
        return {"ok": False, "status": exc.code, "payload": body_txt[:2000]}
    except URLError as exc:
        return {"ok": False, "status": "url_error", "payload": str(exc)}
    except Exception as exc:
        return {"ok": False, "status": "error", "payload": str(exc)}


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    missing = "ui_contract_missing"
    tests = [
        ("GET", f"{base}/yolo/runs/{missing}/summary", None, {400, 404}),
        ("GET", f"{base}/rfdetr/runs/{missing}/summary", None, {400, 404}),
        ("GET", f"{base}/yolo/runs/{missing}/download", None, {400, 404}),
        ("GET", f"{base}/rfdetr/runs/{missing}/download", None, {400, 404}),
        ("DELETE", f"{base}/yolo/runs/{missing}", None, {400, 404}),
        ("DELETE", f"{base}/rfdetr/runs/{missing}", None, {400, 404}),
        ("DELETE", f"{base}/sam3/storage/runs/{missing}", None, {400, 404}),
        ("POST", f"{base}/sam3/storage/runs/{missing}/promote", None, {400, 404}),
        ("GET", f"{base}/datasets/{missing}/check", None, {400, 404}),
        ("GET", f"{base}/datasets/{missing}/download", None, {400, 404}),
        ("DELETE", f"{base}/datasets/{missing}", None, {400, 404}),
        ("POST", f"{base}/datasets/{missing}/build/qwen", None, {400, 404}),
        ("DELETE", f"{base}/qwen/datasets/{missing}", None, {400, 404}),
        ("GET", f"{base}/sam3/datasets/{missing}/classes", None, {400, 404}),
        ("POST", f"{base}/sam3/datasets/{missing}/convert", None, {400, 404}),
        ("POST", f"{base}/prepass/recipes/{missing}/export", None, {400, 404}),
        ("POST", f"{base}/fs/upload_classifier", None, {400, 422}),
        ("POST", f"{base}/fs/upload_labelmap", None, {400, 422}),
    ]

    failures = {}
    for method, url, body, ok in tests:
        result = _safe(f"{method} {url}", method, url, body=body, ok_statuses=ok)
        if not result["ok"]:
            failures[f"{method} {url}"] = result

    summary = {
        "base": base,
        "tested": len(tests),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
