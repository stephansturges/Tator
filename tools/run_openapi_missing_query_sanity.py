#!/usr/bin/env python3
import argparse
import json
import urllib.error
import urllib.request


SKIP_PATH_SUBSTRINGS = (
    "/predict",
    "/sam",
    "/agent_mining/apply",
    "/agent_mining/jobs",
    "/agent_mining/cascades",
    "/agent_mining/recipes",
    "/qwen/caption",
    "/qwen/infer",
    "/qwen/train",
    "/clip/train",
    "/yolo/train",
    "/rfdetr/train",
    "/calibration/jobs",
    "/prepass/jobs",
    "/head_graft/run",
)


def _request(method: str, url: str, body=None, timeout: int = 15):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except urllib.error.URLError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.base_url}/openapi.json") as resp:
        spec = json.load(resp)

    failures = []
    tested = 0

    for path, methods in sorted(spec.get("paths", {}).items()):
        if any(token in path for token in SKIP_PATH_SUBSTRINGS):
            continue
        if "{" in path:
            continue
        for method, meta in methods.items():
            method_upper = method.upper()
            if method_upper not in {"GET", "POST"}:
                continue
            params = meta.get("parameters") or []
            required_queries = [
                p for p in params if p.get("in") == "query" and p.get("required")
            ]
            if not required_queries:
                continue
            url = args.base_url + path
            status = _request(method_upper, url, body=None)
            tested += 1
            if status is None:
                failures.append({"method": method_upper, "path": path, "status": "url_error"})
                continue
            if status >= 500:
                failures.append({"method": method_upper, "path": path, "status": status})

    summary = {
        "base": args.base_url,
        "tested": tested,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
