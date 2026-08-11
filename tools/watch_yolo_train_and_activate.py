#!/usr/bin/env python3
import argparse
import json
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a YOLO training job and activate it when it completes."
    )
    parser.add_argument("job_id", help="YOLO training job id to watch.")
    parser.add_argument(
        "api_root",
        nargs="?",
        default="http://127.0.0.1:8000",
        help="Backend API root. Defaults to http://127.0.0.1:8000.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    job_id = args.job_id
    api_root = args.api_root.rstrip("/")
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
