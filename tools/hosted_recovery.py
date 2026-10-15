#!/usr/bin/env python3
"""Operator-only recovery CLI. Does not contact or stop a running server."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.hosted.artifacts import ArtifactStore
from services.hosted.backup import AgeTool, create_encrypted_backup, restore_encrypted_backup


def callback(value: str):
    module_name, separator, function_name = value.partition(":")
    if not separator or not module_name or not function_name.isidentifier():
        raise ValueError("Callback must be a trusted installed module:function.")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise ValueError("The named callback is not callable.")
    return function


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age", default="age", help="Installed age executable (required for recovery archives)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-age", help="Check the external encryption dependency without reading datasets")
    backup = commands.add_parser("backup", help="Create an encrypted consistent recovery archive")
    backup.add_argument("--database", type=Path, required=True)
    backup.add_argument("--artifacts", type=Path, required=True)
    backup.add_argument("--output", type=Path, required=True)
    backup.add_argument("--recipient", required=True, help="Public age recipient; keep private identity separately on the Mac")
    backup.add_argument("--references", default="services.hosted.operations:recovery_artifact_references", help="Trusted operator override of the standard hosted artifact enumerator (module:function)")
    restore = commands.add_parser("restore", help="Restore into an absent or empty root while the destination server is stopped")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--identity", type=Path, required=True)
    restore.add_argument("--database-name", default="hosted.sqlite3")
    restore.add_argument("--restore-hook", default="services.hosted.operations:prepare_restored_database", help="Trusted operator override of hosted schema validation and authority reset (module:function)")
    args = parser.parse_args(argv)
    age = AgeTool(args.age)
    try:
        if args.command == "check-age":
            result = {"age_version": age.check_available()}
        elif args.command == "backup":
            age.check_available()
            result = dataclasses.asdict(create_encrypted_backup(
                args.database, ArtifactStore(args.artifacts), args.output,
                recipient=args.recipient, references=callback(args.references), age=age,
            ))
        else:
            result = dataclasses.asdict(restore_encrypted_backup(
                args.archive, args.destination, identity=args.identity,
                restore_hook=callback(args.restore_hook), age=age,
                database_name=args.database_name,
            ))
        print(json.dumps(result, default=str, sort_keys=True))
        return 0
    except Exception as error:
        print(f"Recovery operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
