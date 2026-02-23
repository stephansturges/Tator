#!/usr/bin/env python3
"""Guardrail for memory.md append-only updates.

Usage:
  1) Capture a checkpoint at the start of work:
       python tools/memory_guard.py checkpoint
  2) Verify at the end of work that memory.md was appended:
       python tools/memory_guard.py verify --update-checkpoint

This fails (exit code 1) when no new interaction-log entry was appended.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ENTRY_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) \| ")
HEADER_RE = re.compile(r"^- Last updated \(UTC\): (.+)$")


def _parse_iso_utc(value: str) -> datetime:
    raw = str(value or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class MemorySnapshot:
    memory_path: str
    line_count: int
    entry_count: int
    last_entry_ts: Optional[str]
    header_last_updated_ts: Optional[str]
    captured_at: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "memory_path": self.memory_path,
            "line_count": self.line_count,
            "entry_count": self.entry_count,
            "last_entry_ts": self.last_entry_ts,
            "header_last_updated_ts": self.header_last_updated_ts,
            "captured_at": self.captured_at,
        }


def _read_snapshot(memory_path: Path) -> MemorySnapshot:
    if not memory_path.exists():
        raise SystemExit(f"memory file missing: {memory_path}")

    lines = memory_path.read_text(encoding="utf-8").splitlines()
    entry_count = 0
    last_entry_ts: Optional[str] = None
    header_ts: Optional[str] = None

    for line in lines:
        m_header = HEADER_RE.match(line)
        if m_header:
            candidate = m_header.group(1).strip()
            try:
                _parse_iso_utc(candidate)
            except Exception:
                candidate = None
            if candidate:
                header_ts = candidate
            continue

        m_entry = ENTRY_RE.match(line)
        if m_entry:
            entry_count += 1
            last_entry_ts = m_entry.group(1)

    return MemorySnapshot(
        memory_path=str(memory_path.resolve()),
        line_count=len(lines),
        entry_count=entry_count,
        last_entry_ts=last_entry_ts,
        header_last_updated_ts=header_ts,
        captured_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _write_checkpoint(checkpoint_path: Path, snapshot: MemorySnapshot) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(snapshot.to_json(), indent=2), encoding="utf-8")


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise SystemExit(
            f"checkpoint missing: {checkpoint_path}\n"
            "Run: python tools/memory_guard.py checkpoint"
        )
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def _verify(
    before: Dict[str, Any],
    after: MemorySnapshot,
    *,
    max_age_minutes: Optional[float],
    require_header_sync: bool,
) -> Dict[str, Any]:
    errors = []

    before_entries = int(before.get("entry_count") or 0)
    if after.entry_count <= before_entries:
        errors.append(
            f"no appended interaction entries: before={before_entries} after={after.entry_count}"
        )

    before_last = before.get("last_entry_ts")
    if before_last and after.last_entry_ts:
        try:
            if _parse_iso_utc(after.last_entry_ts) <= _parse_iso_utc(str(before_last)):
                errors.append(
                    f"last entry timestamp did not advance: before={before_last} after={after.last_entry_ts}"
                )
        except Exception:
            errors.append("invalid last entry timestamp format")

    if require_header_sync:
        if not after.header_last_updated_ts:
            errors.append("missing/invalid 'Last updated (UTC)' header")
        if after.last_entry_ts and after.header_last_updated_ts:
            try:
                if _parse_iso_utc(after.header_last_updated_ts) < _parse_iso_utc(after.last_entry_ts):
                    errors.append(
                        f"header timestamp older than last entry: header={after.header_last_updated_ts} entry={after.last_entry_ts}"
                    )
            except Exception:
                errors.append("invalid header timestamp format")

    if max_age_minutes is not None and after.last_entry_ts:
        age = (
            datetime.now(timezone.utc) - _parse_iso_utc(after.last_entry_ts)
        ).total_seconds() / 60.0
        if age > float(max_age_minutes):
            errors.append(
                f"latest memory entry is stale: age={age:.2f}m max={float(max_age_minutes):.2f}m"
            )

    return {
        "ok": not errors,
        "errors": errors,
        "before": before,
        "after": after.to_json(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Append-only guard for memory.md updates.")
    parser.add_argument("--memory", default="memory.md", help="Path to memory file.")
    parser.add_argument(
        "--checkpoint",
        default="/tmp/tator_memory_guard_checkpoint.json",
        help="Checkpoint path.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("checkpoint", help="Capture a checkpoint for later verification.")
    sub.add_parser("status", help="Print current memory snapshot.")

    verify = sub.add_parser("verify", help="Verify memory was appended since checkpoint.")
    verify.add_argument(
        "--max-age-minutes",
        type=float,
        default=None,
        help="Optional max age for latest log entry.",
    )
    verify.add_argument(
        "--no-require-header-sync",
        action="store_true",
        help="Skip check that header timestamp is >= latest entry timestamp.",
    )
    verify.add_argument(
        "--update-checkpoint",
        action="store_true",
        help="Overwrite checkpoint with current snapshot after successful verify.",
    )

    args = parser.parse_args()
    memory_path = Path(args.memory)
    checkpoint_path = Path(args.checkpoint)

    current = _read_snapshot(memory_path)

    if args.cmd == "status":
        print(json.dumps(current.to_json(), indent=2))
        return

    if args.cmd == "checkpoint":
        _write_checkpoint(checkpoint_path, current)
        print(json.dumps({"ok": True, "checkpoint": str(checkpoint_path), "snapshot": current.to_json()}, indent=2))
        return

    if args.cmd == "verify":
        before = _load_checkpoint(checkpoint_path)
        result = _verify(
            before,
            current,
            max_age_minutes=args.max_age_minutes,
            require_header_sync=not bool(args.no_require_header_sync),
        )
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            sys.exit(1)
        if args.update_checkpoint:
            _write_checkpoint(checkpoint_path, current)
        return

    raise SystemExit(f"unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
