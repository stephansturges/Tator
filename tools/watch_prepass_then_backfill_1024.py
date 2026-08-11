#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _count_cache_images(cache_key: str) -> Tuple[int, float]:
    cache_dir = Path("uploads/calibration_cache/prepass") / str(cache_key).strip() / "images"
    if not cache_dir.exists():
        return 0, 0.0
    count = 0
    newest = 0.0
    for entry in cache_dir.glob("*.json"):
        count += 1
        try:
            newest = max(newest, float(entry.stat().st_mtime))
        except Exception:
            continue
    return count, newest


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait until prepass cache keys reach target image counts, then run one-time 1024-d backfill."
    )
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument("--classifier-id", required=True, help="Classifier id/path.")
    parser.add_argument("--cache-key", action="append", required=True, help="Prepass cache key (repeatable).")
    parser.add_argument(
        "--target-images",
        type=int,
        default=4000,
        help="Required minimum image records per cache key before backfill runs.",
    )
    parser.add_argument(
        "--stable-seconds",
        type=int,
        default=180,
        help="Require newest cache file to be at least this old before starting backfill.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Pass --force to backfill script.")
    parser.add_argument(
        "--log",
        default=None,
        help="Optional log file path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    target = max(1, int(args.target_images))
    stable = max(0, int(args.stable_seconds))
    poll = max(5, int(args.poll_seconds))
    cache_keys = [str(key).strip() for key in args.cache_key if str(key).strip()]
    if not cache_keys:
        raise SystemExit("No --cache-key values provided.")

    log_path = Path(args.log).resolve() if args.log else None

    def _log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}"
        if log_path is None:
            print(line, flush=True)
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    _log(
        "watch_start "
        + json.dumps(
            {
                "dataset": args.dataset,
                "cache_keys": cache_keys,
                "target_images": target,
                "stable_seconds": stable,
            },
            sort_keys=True,
        )
    )
    while True:
        now = time.time()
        status: Dict[str, Dict[str, float]] = {}
        ready = True
        for key in cache_keys:
            count, newest = _count_cache_images(key)
            age = (now - newest) if newest > 0 else 0.0
            status[key] = {"count": float(count), "newest_age": float(age)}
            if count < target:
                ready = False
                continue
            if stable > 0 and newest > 0 and age < stable:
                ready = False
        _log("watch_tick " + json.dumps(status, sort_keys=True))
        if ready:
            break
        time.sleep(poll)

    cmd: List[str] = [
        sys.executable,
        "tools/backfill_prepass_embeddings.py",
        "--dataset",
        args.dataset,
        "--classifier-id",
        args.classifier_id,
        "--embed-proj-dims",
        "1024",
        "--device",
        args.device,
    ]
    if args.force:
        cmd.append("--force")
    for key in cache_keys:
        cmd.extend(["--cache-key", key])

    _log("backfill_start " + json.dumps({"cmd": cmd}, sort_keys=True))
    _run(cmd)
    _log("backfill_done")


if __name__ == "__main__":
    main()
