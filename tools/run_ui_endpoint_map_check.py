#!/usr/bin/env python3
import argparse
import json
import re
import urllib.request
from pathlib import Path


API_PATTERN = re.compile(r"\$\{API_ROOT\}(/[^\"'\\]+)")
ENCODE_SEGMENT = re.compile(r"/\\$\\{encodeURIComponent\\([^}]+\\)\\}")


def _normalize_path(raw: str) -> str:
    path = raw.split("?")[0]
    path = path.split("`")[0]
    path = ENCODE_SEGMENT.sub("/{param}", path)
    path = re.sub(r"/\\$\\{[^}]+\\}", "/{param}", path)
    return path


def _extract_paths(js_text: str):
    paths = set()
    for match in API_PATTERN.finditer(js_text):
        raw = match.group(1)
        if not raw:
            continue
        if "${endpoint}" in raw:
            continue
        norm = _normalize_path(raw)
        if norm:
            paths.add(norm)
    return paths


def _match_openapi_path(ui_path: str, openapi_paths: set[str]) -> bool:
    if ui_path in openapi_paths:
        return True
    # normalize both by treating {param} as wildcard
    ui_parts = ui_path.strip("/").split("/")
    for candidate in openapi_paths:
        cand_parts = candidate.strip("/").split("/")
        if len(ui_parts) != len(cand_parts):
            continue
        ok = True
        for u, c in zip(ui_parts, cand_parts):
            if c.startswith("{") and c.endswith("}"):
                continue
            if u.startswith("{") and u.endswith("}"):
                continue
            if u != c:
                ok = False
                break
        if ok:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--js", default="ybat-master/ybat.js")
    args = parser.parse_args()

    js_path = Path(args.js)
    js_text = js_path.read_text(encoding="utf-8")
    ui_paths = sorted(_extract_paths(js_text))

    with urllib.request.urlopen(f"{args.base_url}/openapi.json") as resp:
        spec = json.load(resp)
    openapi_paths = set(spec.get("paths", {}).keys())

    missing = [path for path in ui_paths if not _match_openapi_path(path, openapi_paths)]

    payload = {
        "ui_paths": len(ui_paths),
        "missing": missing,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
