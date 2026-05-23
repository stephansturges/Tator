#!/usr/bin/env python3
import argparse
import json
import re
import urllib.request
from pathlib import Path


FETCH_RE = re.compile(
    r"fetch\(\s*`\$\{API_ROOT\}(/[^`]+)`\s*(?:,\s*\{([\s\S]*?)\})?\s*\)",
    re.MULTILINE,
)
METHOD_RE = re.compile(r"method\s*:\s*['\"](GET|POST|PUT|PATCH|DELETE)['\"]", re.IGNORECASE)
ENCODE_SEGMENT = re.compile(r"/\$\{encodeURIComponent\([^}]+\)\}")


def _normalize_path(raw: str) -> str:
    path = raw.split("?")[0]
    path = ENCODE_SEGMENT.sub("/{param}", path)
    path = re.sub(r"/\$\{[^}]+\}", "/{param}", path)
    return path


def _extract_fetches(js_text: str):
    entries = []
    for match in FETCH_RE.finditer(js_text):
        raw_path = match.group(1)
        if "${endpoint}" in raw_path:
            continue
        method = "GET"
        # Heuristic: scan a window of text following the match for method overrides.
        scan_start = match.start()
        scan_end = min(len(js_text), match.end() + 400)
        scan_text = js_text[scan_start:scan_end]
        method_match = METHOD_RE.search(scan_text)
        if method_match:
            method = method_match.group(1).upper()
        entries.append((method, _normalize_path(raw_path)))
    return entries


def _path_match_score(ui_parts: list[str], cand_parts: list[str]) -> int | None:
    score = 0
    for u, c in zip(ui_parts, cand_parts, strict=False):
        ui_param = u.startswith("{") and u.endswith("}")
        candidate_param = c.startswith("{") and c.endswith("}")
        if ui_param:
            if not candidate_param:
                return None
            score += 1
        elif candidate_param:
            score += 1
        elif u == c:
            score += 3
        else:
            return None
    return score


def _match_openapi_path(ui_path: str, openapi_paths: set[str]) -> str | None:
    if ui_path in openapi_paths:
        return ui_path
    ui_parts = ui_path.strip("/").split("/")
    ui_has_param = any(part.startswith("{") and part.endswith("}") for part in ui_parts)
    # Prefer parametric matches when UI uses params.
    candidates = list(openapi_paths)
    if ui_has_param:
        candidates.sort(key=lambda p: ("{" not in p or "}" not in p))
    best_path = None
    best_score = -1
    for candidate in candidates:
        cand_parts = candidate.strip("/").split("/")
        if len(ui_parts) != len(cand_parts):
            continue
        score = _path_match_score(ui_parts, cand_parts)
        if score is not None and score > best_score:
            best_path = candidate
            best_score = score
    return best_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--js", default="ybat-master/ybat.js")
    args = parser.parse_args()

    js_text = Path(args.js).read_text(encoding="utf-8")
    ui_entries = _extract_fetches(js_text)

    with urllib.request.urlopen(f"{args.base_url}/openapi.json") as resp:
        spec = json.load(resp)
    openapi_paths = spec.get("paths", {})

    failures = []
    for method, path in ui_entries:
        match_path = _match_openapi_path(path, set(openapi_paths.keys()))
        if not match_path:
            failures.append({"method": method, "path": path, "error": "missing_path"})
            continue
        available_methods = set(openapi_paths[match_path].keys())
        if method.lower() not in available_methods:
            failures.append(
                {
                    "method": method,
                    "path": path,
                    "error": "missing_method",
                    "available": sorted(available_methods),
                }
            )

    payload = {
        "ui_fetches": len(ui_entries),
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
