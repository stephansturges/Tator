#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path
from urllib import request

def normalize(path: str) -> str:
    path = path.split("?", 1)[0]
    path = re.sub(r"\$\{[^}]+\}", "{}", path)
    path = re.sub(r"\{[^}]+\}", "{}", path)
    path = re.sub(r"//+", "/", path)
    if not path.startswith("/"):
        path = "/" + path
    return path


def extract_ui_endpoints(js_text: str) -> list[tuple[str, str]]:
    endpoints: list[tuple[str, str]] = []
    # Capture fetch(`${API_ROOT}/...` ...)
    fetch_re = re.compile(r"fetch\(\s*`\$\{API_ROOT\}/([^`]+)`(?P<rest>[^;]{0,400})", re.S)
    for match in fetch_re.finditer(js_text):
        path = match.group(1)
        rest = match.group("rest") or ""
        method = "GET"
        method_m = re.search(r"method\s*:\s*[\"']([A-Z]+)[\"']", rest)
        if method_m:
            method = method_m.group(1).upper()
        endpoints.append((normalize(path), method))

    # Deduplicate while preserving order
    seen = set()
    out: list[tuple[str, str]] = []
    for path, method in endpoints:
        key = (path, method)
        if key in seen:
            continue
        seen.add(key)
        out.append((path, method))
    return out


def load_openapi(base_url: str) -> dict:
    with request.urlopen(f"{base_url}/openapi.json") as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    js_path = Path("ybat-master/ybat.js")
    if not js_path.exists():
        print("ybat.js not found")
        return 1

    js_text = js_path.read_text(encoding="utf-8", errors="ignore")
    ui_endpoints = extract_ui_endpoints(js_text)

    openapi = load_openapi(base_url)
    paths = openapi.get("paths", {})
    openapi_methods = {}
    for path, ops in paths.items():
        norm = normalize(path)
        methods = {m.upper() for m in ops.keys()}
        openapi_methods[norm] = methods

    missing = []
    method_mismatch = []
    for path, method in ui_endpoints:
        if path not in openapi_methods:
            missing.append(path)
            continue
        if method != "ANY" and method not in openapi_methods[path]:
            method_mismatch.append((path, method, sorted(openapi_methods[path])))

    print("UI endpoints found:", len(ui_endpoints))
    print("OpenAPI paths:", len(openapi_methods))

    if missing:
        print("\nMissing paths (UI -> OpenAPI):")
        for p in sorted(set(missing)):
            print("-", p)
    else:
        print("\nNo missing paths detected.")

    if method_mismatch:
        print("\nMethod mismatches:")
        for p, m, have in method_mismatch:
            print(f"- {p}: UI expects {m}, OpenAPI has {have}")
    else:
        print("\nNo method mismatches detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
