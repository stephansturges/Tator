#!/usr/bin/env python3
import argparse
import json
import os
import uuid
from urllib import request, error


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _get(base_url: str, path: str):
    req = request.Request(f"{base_url}{path}", headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(base_url: str, path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _delete(base_url: str, path: str):
    req = request.Request(f"{base_url}{path}", method="DELETE")
    with request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _safe(_label, fn):
    try:
        return {"ok": True, "result": fn()}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        return {"ok": False, "error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UI data-operation smoke checks against a Tator backend."
    )
    parser.add_argument(
        "base_url",
        nargs="?",
        default=None,
        help=f"Backend base URL. Defaults to BASE_URL or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url_flag",
        default=None,
        help="Backend base URL. Overrides the positional URL and BASE_URL.",
    )
    args = parser.parse_args(argv)
    args.base_url = (
        args.base_url_flag
        or args.base_url
        or os.environ.get("BASE_URL", DEFAULT_BASE_URL)
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_url = args.base_url.rstrip("/")
    results = {}

    # Glossary create/get/delete
    gloss_name = f"test_glossary_{uuid.uuid4().hex[:6]}"
    gloss_text = "test_label: item, object\n"
    results["glossary_create"] = _safe(
        "glossary_create",
        lambda: _post(
            base_url,
            "/glossaries",
            {"name": gloss_name, "glossary": gloss_text},
        ),
    )
    results["glossary_get"] = _safe(
        "glossary_get", lambda: _get(base_url, f"/glossaries/{gloss_name}")
    )
    results["glossary_delete"] = _safe(
        "glossary_delete", lambda: _delete(base_url, f"/glossaries/{gloss_name}")
    )

    # Prepass recipe create/get/delete
    recipe_id = f"test_recipe_{uuid.uuid4().hex[:6]}"
    recipe_payload = {
        "recipe_id": recipe_id,
        "name": recipe_id,
        "description": "ui data ops test",
        "config": {"enable_yolo": True, "enable_rfdetr": True},
        "glossary": gloss_text,
    }
    results["prepass_recipe_create"] = _safe(
        "prepass_recipe_create",
        lambda: _post(base_url, "/prepass/recipes", recipe_payload),
    )
    results["prepass_recipe_get"] = _safe(
        "prepass_recipe_get", lambda: _get(base_url, f"/prepass/recipes/{recipe_id}")
    )
    results["prepass_recipe_delete"] = _safe(
        "prepass_recipe_delete", lambda: _delete(base_url, f"/prepass/recipes/{recipe_id}")
    )

    # Dataset glossary set/restore
    datasets = _safe("datasets", lambda: _get(base_url, "/datasets"))
    results["datasets"] = datasets
    dataset_id = None
    if datasets.get("ok"):
        payload = datasets.get("result")
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                dataset_id = first.get("id") or first.get("dataset_id")
            elif isinstance(first, str):
                dataset_id = first
    if dataset_id:
        prev_glossary = _safe(
            "dataset_glossary_get",
            lambda: _get(base_url, f"/datasets/{dataset_id}/glossary"),
        )
        results["dataset_glossary_get"] = prev_glossary
        results["dataset_glossary_set"] = _safe(
            "dataset_glossary_set",
            lambda: _post(
                base_url,
                f"/datasets/{dataset_id}/glossary",
                {"glossary": gloss_text},
            ),
        )
        # Restore
        restore_text = ""
        if prev_glossary.get("ok"):
            prev_payload = prev_glossary.get("result") or {}
            if isinstance(prev_payload, dict):
                restore_text = str(prev_payload.get("glossary") or "")
            elif isinstance(prev_payload, str):
                restore_text = prev_payload
        results["dataset_glossary_restore"] = _safe(
            "dataset_glossary_restore",
            lambda: _post(
                base_url,
                f"/datasets/{dataset_id}/glossary",
                {"glossary": restore_text},
            ),
        )

    print(json.dumps({"base_url": base_url, "results": results}, indent=2))
    failures = {k: v for k, v in results.items() if not v.get("ok")}
    if failures:
        print("\nFailures:")
        print(json.dumps(failures, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
