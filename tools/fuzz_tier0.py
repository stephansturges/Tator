#!/usr/bin/env python3
import argparse
import json
import urllib.request
from urllib.error import URLError, HTTPError


def _get(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error for {url}: {exc}") from exc
    return json.loads(data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tier-0 backend smoke checks against a Tator backend."
    )
    parser.add_argument(
        "base_url",
        nargs="?",
        default="http://127.0.0.1:8000",
        help="Backend base URL. Defaults to http://127.0.0.1:8000.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = args.base_url.rstrip("/")
    results = {}
    timeouts = {
        "/system/health_summary": 30,
        "/clip/classifiers": 30,
    }
    for path in (
        "/system/health_summary",
        "/system/storage_check",
        "/system/gpu",
        "/sam3/models/available",
        "/qwen/models",
        "/detectors/default",
        "/clip/classifiers",
    ):
        results[path] = _get(base + path, timeout=timeouts.get(path, 15))

    # Minimal schema checks
    health = results["/system/health_summary"]
    assert ("ok" in health) or ("status" in health), "health_summary missing ok/status"
    det_default = results["/detectors/default"]
    assert ("active" in det_default) or ("mode" in det_default), "/detectors/default missing active/mode"
    classifiers_payload = results["/clip/classifiers"]
    assert (
        isinstance(classifiers_payload, list)
        or ("classifiers" in classifiers_payload)
    ), "/clip/classifiers missing classifiers"

    print(json.dumps({"tier0": "ok", "base": base}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
