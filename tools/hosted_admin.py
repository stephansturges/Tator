"""Local operator account bootstrap; passwords never appear in command arguments."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from services.hosted.auth import AuthService
from services.hosted.store import HostedStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=os.environ.get("TATOR_HOSTED_STATE_DIR"), required=not bool(os.environ.get("TATOR_HOSTED_STATE_DIR")))
    parser.add_argument("username")
    args = parser.parse_args()
    password = getpass.getpass("Administrator password (at least 12 characters): ")
    if password != getpass.getpass("Repeat password: "):
        parser.error("Passwords do not match.")
    service = AuthService(HostedStore(Path(args.state_dir) / "hosted.sqlite3"))
    user = service.bootstrap(args.username, password)
    print(f"Administrator {user['username']} created. Open the configured hosted URL to log in.")


if __name__ == "__main__":
    main()
