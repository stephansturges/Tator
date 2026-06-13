#!/usr/bin/env python3
"""Simple concurrency smoke test for safe GET endpoints."""
import argparse
import concurrent.futures
import json
import os
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
ENDPOINTS = [
    "/system/gpu",
    "/system/storage_check",
    "/datasets",
    "/glossaries",
    "/clip/classifiers",
    "/yolo/active",
    "/rfdetr/active",
    "/qwen/status",
]


def fetch(base_url: str, path: str):
    with urllib.request.urlopen(f"{base_url}{path}", timeout=20) as resp:
        return path, resp.getcode(), json.loads(resp.read().decode("utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run concurrent safe-GET UI smoke checks against a Tator backend."
    )
    parser.add_argument(
        "base_url",
        nargs="?",
        default=None,
        help=f"Backend base URL. Defaults to BASE_URL or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url_flag",
        default=None,
        help="Backend base URL. Overrides the positional URL and BASE_URL.",
    )
    args = parser.parse_args(argv)
    args.base_url = (
        args.base_url_flag
        or args.base_url
        or os.environ.get("BASE_URL", DEFAULT_BASE_URL)
    )
    return args


def run_smoke(base_url: str) -> dict:
    normalized_base_url = base_url.rstrip("/")
    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
        futures = [exe.submit(fetch, normalized_base_url, path) for path in ENDPOINTS]
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                failures.append(str(exc))
    return {
        "base_url": normalized_base_url,
        "failures": failures,
        "results": [
            (path, code) for path, code, _ in sorted(results, key=lambda row: row[0])
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_smoke(args.base_url)
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
