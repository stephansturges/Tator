#!/usr/bin/env python3
"""Simple concurrency smoke test for safe GET endpoints."""
import json
import concurrent.futures
import urllib.request

BASE_URL = "http://127.0.0.1:8000"
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


def fetch(path: str):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=20) as resp:
        return path, resp.getcode(), json.loads(resp.read().decode("utf-8"))


results = []
failures = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
    futures = [exe.submit(fetch, path) for path in ENDPOINTS]
    for fut in concurrent.futures.as_completed(futures):
        try:
            results.append(fut.result())
        except Exception as exc:
            failures.append(str(exc))

print(json.dumps({"failures": failures, "results": [(p, code) for p, code, _ in results]}, indent=2))
