from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ui_env_module():
    path = REPO_ROOT / "tests" / "ui" / "e2e" / "helpers" / "env.py"
    spec = importlib.util.spec_from_file_location("tator_ui_e2e_env", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["tator_ui_e2e_env"] = module
    spec.loader.exec_module(module)
    return module


def test_require_ui_env_defaults_to_backend_served_ui_and_fixture(monkeypatch) -> None:
    ui_env = _load_ui_env_module()
    monkeypatch.delenv("UI_PAGE_URL", raising=False)
    monkeypatch.delenv("UI_DATASET_PATH", raising=False)
    monkeypatch.setenv("UI_DATASET_STAGE", "0")
    monkeypatch.setenv("UI_API_ROOT", "http://127.0.0.1:8123")

    page_url, dataset_path = ui_env.require_ui_env()

    assert page_url == "http://127.0.0.1:8123/tator.html"
    assert Path(dataset_path).samefile(ui_env._repo_root() / "tests" / "fixtures" / "fuzz_pack")


def test_require_ui_env_preserves_explicit_overrides(monkeypatch, tmp_path: Path) -> None:
    ui_env = _load_ui_env_module()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    monkeypatch.setenv("UI_PAGE_URL", "http://example.test/custom.html")
    monkeypatch.setenv("UI_DATASET_PATH", str(dataset_root))
    monkeypatch.setenv("UI_DATASET_STAGE", "0")

    page_url, dataset_path = ui_env.require_ui_env()

    assert page_url == "http://example.test/custom.html"
    assert Path(dataset_path) == dataset_root.resolve()
