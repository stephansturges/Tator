#!/usr/bin/env python3
import json
import sys
import urllib.request
from urllib.error import URLError, HTTPError


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error for {url}: {exc}") from exc
    return json.loads(data)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    results = {}
    for path in (
        "/system/health_summary",
        "/datasets",
        "/qwen/datasets",
        "/sam3/models/available",
        "/qwen/models",
        "/detectors/default",
        "/clip/classifiers",
    ):
        results[path] = _get(base + path)

    # Minimal schema checks
    assert "status" in results["/system/health_summary"], "health_summary missing status"
    assert isinstance(results["/datasets"], list), "/datasets must be list"
    assert isinstance(results["/qwen/datasets"], list), "/qwen/datasets must be list"
    assert "active" in results["/detectors/default"], "/detectors/default missing active"
    assert "classifiers" in results["/clip/classifiers"], "/clip/classifiers missing classifiers"

    print(json.dumps({"tier0": "ok", "base": base}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
