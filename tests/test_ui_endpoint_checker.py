from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_checker_module():
    path = REPO_ROOT / "tools" / "check_ui_endpoints.py"
    spec = importlib.util.spec_from_file_location("check_ui_endpoints", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_ui_endpoints_expands_detector_endpoint_variable() -> None:
    checker = _load_checker_module()
    js = """
    const endpoint = mode === "yolo" ? "yolo" : "rfdetr";
    fetch(`${API_ROOT}/${endpoint}/runs/${encodeURIComponent(entry.run_id)}`, { method: "DELETE" });
    fetch(`${API_ROOT}/${endpoint}/runs/${encodeURIComponent(entry.run_id)}/download`);
    """

    endpoints = set(checker.extract_ui_endpoints(js))

    assert ("/yolo/runs/{}", "DELETE") in endpoints
    assert ("/rfdetr/runs/{}", "DELETE") in endpoints
    assert ("/yolo/runs/{}/download", "GET") in endpoints
    assert ("/rfdetr/runs/{}/download", "GET") in endpoints
    assert ("/{}/runs/{}", "DELETE") not in endpoints
