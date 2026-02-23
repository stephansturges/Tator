#!/usr/bin/env python3
import json
import sys
import time
from urllib import request


def fetch(url: str) -> dict:
    with request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post(url: str, payload: dict) -> dict:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: watch_yolo_train_and_activate.py <job_id> [api_root]")
        return 2
    job_id = sys.argv[1]
    api_root = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
    job_url = f"{api_root}/yolo/train/jobs/{job_id}"
    active_url = f"{api_root}/yolo/active"

    while True:
        job = fetch(job_url)
        status = job.get("status")
        progress = job.get("progress")
        message = job.get("message")
        if status in {"failed", "cancelled"}:
            print(f"{job_id} {status}: {job.get('error') or message}")
            return 1
        if status in {"done", "succeeded", "complete"}:
            # Promote this run id to active.
            post(active_url, {"run_id": job_id})
            print(f"{job_id} activated")
            return 0
        print(f"{job_id} {status} {progress}")
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
