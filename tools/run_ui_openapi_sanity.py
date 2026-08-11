#!/usr/bin/env python3
import argparse
import json
import re
import sys
import urllib.error
import urllib.request


SKIP_POST_PATHS = {
    "/calibration/jobs",
    "/prepass/jobs",
    "/yolo/train/jobs",
    "/rfdetr/train/jobs",
    "/clip/train",
    "/qwen/train",
    "/clip/active_model",
    "/glossaries",
    "/qwen/dataset/init",
    "/qwen/prepass",
    "/qwen/settings",
}

SKIP_SUBSTRINGS = (
    "/predict",
    "/sam",
    "/agent_mining/apply",
    "/agent_mining/jobs",
    "/agent_mining/cascades",
    "/agent_mining/recipes",
    "/qwen/caption",
    "/qwen/infer",
    "/qwen/infer_window",
    "/head_graft/run",
)


def _fetch_json(url, method="GET", payload=None, headers=None, timeout=15):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers=headers or {"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return None, str(exc)


def _replace_path_params(path: str) -> str:
    return re.sub(r"\{[^/]+\}", "missing", path)


def _should_skip(method: str, path: str, request_body_required: bool) -> bool:
    if method in {"POST", "PUT", "PATCH"}:
        if path in SKIP_POST_PATHS:
            return True
        if any(token in path for token in SKIP_SUBSTRINGS):
            return True
        if not request_body_required:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    status_code, body = _fetch_json(f"{args.base_url}/openapi.json", timeout=20)
    if status_code != 200:
        print(json.dumps({"error": "failed_openapi", "status": status_code, "body": body}))
        return 1
    spec = json.loads(body)
    paths = spec.get("paths", {})

    results = []
    skipped = []
    failures = []

    for path, methods in sorted(paths.items()):
        for method, meta in methods.items():
            method = method.upper()
            request_body_required = False
            if "requestBody" in meta:
                rb = meta["requestBody"] or {}
                request_body_required = bool(rb.get("required")) or bool(rb.get("content"))

            if _should_skip(method, path, request_body_required):
                skipped.append(f"{method} {path}")
                continue

            url = args.base_url + _replace_path_params(path)
            if method in {"GET", "DELETE"}:
                status, _ = _fetch_json(url, method=method, payload=None, timeout=15)
                ok = status in {200, 400, 404, 405, 422}
            else:
                status, _ = _fetch_json(url, method=method, payload={}, timeout=15)
                ok = status in {400, 404, 405, 422}

            results.append({"method": method, "path": path, "status": status})
            if not ok:
                failures.append({"method": method, "path": path, "status": status})

    payload = {
        "base": args.base_url,
        "tested": len(results),
        "skipped": len(skipped),
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    if failures:
        print("\nSkipped endpoints:")
        for entry in skipped:
            print(entry)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
