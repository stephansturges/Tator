#!/usr/bin/env python3
"""Inventory a legacy dataset, then explicitly import its frozen plan."""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.hosted.artifacts import ArtifactStore
from services.hosted.auth import AuthService
from services.hosted.migration import HostedMigrator, inventory, inventory_report
from services.hosted.store import HostedStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inventory", help="Read and hash source files without creating hosted state")
    inspect.add_argument("--source", type=Path, required=True)
    inspect.add_argument("--mapping", type=Path, help="Explicit JSON metadata/artifact preservation mapping")
    inspect.add_argument("--output", type=Path, required=True, help="New plan JSON outside the source tree; also writes a .rst report")
    inspect.add_argument("--clip-boxes", action="store_true", help="Explicitly clamp out-of-bounds source boxes/polygons during conversion")
    apply = commands.add_parser("apply", help="Revalidate a reviewed inventory and resume its hosted import")
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--state-dir", type=Path, required=True)
    apply.add_argument("--name", required=True)
    apply.add_argument("--admin", required=True, help="Existing enabled administrator username; password is prompted")
    apply.add_argument("--mode", choices=("incoming", "approved-baseline"), default="incoming")
    apply.add_argument("--accept-existing-labels", action="store_true", help="Explicitly approve source labels under the current administrator's identity")
    args = parser.parse_args(argv)
    if args.command == "inventory":
        output = args.output.absolute()
        report = output.with_suffix(".rst")
        if output.resolve().is_relative_to(args.source.resolve()):
            parser.error("Write inventory outputs outside the source dataset.")
        if output.exists() or report.exists():
            parser.error("Inventory output already exists; choose a new output path.")
        mapping = json.loads(args.mapping.read_text()) if args.mapping else None
        plan = inventory(args.source, mapping=mapping, clip_boxes=args.clip_boxes)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        with report.open("x") as stream:
            stream.write(inventory_report(plan))
        print(json.dumps({"plan": str(output), "report": str(report), "images": len(plan["images"]), "blockers": len(plan["blockers"]), "plan_sha256": plan["plan_sha256"]}))
    else:
        if not (args.state_dir / "hosted.sqlite3").is_file():
            parser.error("Use an existing hosted installation with an administrator account.")
        plan = json.loads(args.plan.read_text())
        store = HostedStore(args.state_dir / "hosted.sqlite3")
        auth = AuthService(store)
        session = auth.login(args.admin, getpass.getpass("Administrator password: "), address="local-migration-cli")
        actor = auth.authenticate(session.token)
        try:
            result = HostedMigrator(store, ArtifactStore(args.state_dir / "artifacts")).apply(actor, plan, name=args.name, mode=args.mode, accept_existing_labels=args.accept_existing_labels)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        finally:
            auth.logout(session.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
