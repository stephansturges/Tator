from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_ui_contract_dataset_cleanup_removes_only_generated_trash(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool_module("run_ui_contract_tests", "tools/run_ui_contract_tests.py")
    monkeypatch.chdir(tmp_path)
    dataset_id = "ui_contract_upload_123"
    trash_dir = tmp_path / "uploads" / "datasets" / ".trash" / f"{dataset_id}-deleted"
    trash_dir.mkdir(parents=True)
    (trash_dir / "payload.bin").write_bytes(b"generated")

    tool._cleanup_contract_dataset_trash(
        {"status": "trashed", "trash_path": str(trash_dir)},
        dataset_id,
    )

    assert not trash_dir.exists()


def test_ui_contract_dataset_cleanup_keeps_non_contract_trash(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool_module("run_ui_contract_tests", "tools/run_ui_contract_tests.py")
    monkeypatch.chdir(tmp_path)
    dataset_id = "user_dataset"
    trash_dir = tmp_path / "uploads" / "datasets" / ".trash" / f"{dataset_id}-deleted"
    trash_dir.mkdir(parents=True)
    (trash_dir / "payload.bin").write_bytes(b"user")

    tool._cleanup_contract_dataset_trash(
        {"status": "trashed", "trash_path": str(trash_dir)},
        dataset_id,
    )

    assert trash_dir.exists()
    assert (trash_dir / "payload.bin").read_bytes() == b"user"


def test_gpu_validation_cleanup_removes_run_scoped_dataset_trash(tmp_path: Path) -> None:
    tool = _load_tool_module("run_gpu_validation_suite", "tools/run_gpu_validation_suite.py")
    suite = tool.GpuValidationSuite(
        repo_root=tmp_path,
        base_url="http://127.0.0.1:8000",
        timeout_s=5,
        run_id="unit",
        cleanup=True,
    )
    dataset_id = f"{suite.run_id}_dataset"
    trash_dir = tmp_path / "uploads" / "datasets" / ".trash" / f"{dataset_id}-deleted"
    trash_dir.mkdir(parents=True)
    (trash_dir / "payload.bin").write_bytes(b"generated")
    removed: list[str] = []
    skipped: list[str] = []

    suite._cleanup_dataset_trash_payload(
        {"status": "trashed", "trash_path": str(trash_dir)},
        dataset_id,
        removed,
        skipped,
    )

    assert not trash_dir.exists()
    assert str(trash_dir.resolve()) in removed
    assert skipped == []


def test_gpu_validation_cleanup_keeps_unscoped_dataset_trash(tmp_path: Path) -> None:
    tool = _load_tool_module("run_gpu_validation_suite", "tools/run_gpu_validation_suite.py")
    suite = tool.GpuValidationSuite(
        repo_root=tmp_path,
        base_url="http://127.0.0.1:8000",
        timeout_s=5,
        run_id="unit",
        cleanup=True,
    )
    trash_dir = tmp_path / "uploads" / "datasets" / ".trash" / "user_dataset-deleted"
    trash_dir.mkdir(parents=True)
    (trash_dir / "payload.bin").write_bytes(b"user")
    removed: list[str] = []
    skipped: list[str] = []

    suite._cleanup_dataset_trash_payload(
        {"status": "trashed", "trash_path": str(trash_dir)},
        "user_dataset",
        removed,
        skipped,
    )

    assert trash_dir.exists()
    assert removed == []
    assert skipped == [str(trash_dir)]


def test_gpu_validation_safe_remove_rejects_allowed_root_prefix_sibling(
    tmp_path: Path,
) -> None:
    tool = _load_tool_module("run_gpu_validation_suite", "tools/run_gpu_validation_suite.py")
    suite = tool.GpuValidationSuite(
        repo_root=tmp_path,
        base_url="http://127.0.0.1:8000",
        timeout_s=5,
        run_id="unit",
        cleanup=True,
    )
    sibling = tmp_path / "uploads_evil" / "unit_payload"
    sibling.mkdir(parents=True)
    (sibling / "payload.bin").write_bytes(b"outside")
    removed: list[str] = []
    skipped: list[str] = []

    suite._safe_remove_path(sibling, removed, skipped)

    assert sibling.exists()
    assert (sibling / "payload.bin").read_bytes() == b"outside"
    assert removed == []
    assert skipped == [str(sibling.resolve())]


def test_ui_param_sweep_import_is_side_effect_free(monkeypatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("run_ui_param_sweep must not call the backend during import")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    tool = _load_tool_module("run_ui_param_sweep", "tools/run_ui_param_sweep.py")

    assert callable(tool.run_sweep)
    assert callable(tool.main)


def test_ui_param_sweep_accepts_detector_precondition_statuses() -> None:
    tool = _load_tool_module("run_ui_param_sweep_status", "tools/run_ui_param_sweep.py")

    assert 412 in tool.OK_STATUS


def test_qwen_prepass_wrappers_are_executable_and_resolve_repo_python() -> None:
    for rel_path in [
        "tools/run_qwen_prepass_smoke.sh",
        "tools/run_qwen_prepass_benchmark.sh",
    ]:
        path = REPO_ROOT / rel_path
        text = path.read_text(encoding="utf-8")

        assert os.access(path, os.X_OK)
        assert 'PYTHON_BIN="$(resolve_python)"' in text
        assert '"${PYTHON_BIN}"' in text
        assert "\npython " not in text


def test_auto_mlp_runner_has_no_host_specific_root_or_bare_python_calls() -> None:
    text = (REPO_ROOT / "tools/auto_mlp_run.sh").read_text(encoding="utf-8")

    assert "/home/steph/Tator" not in text
    assert "ROOT_DIR=\"${ROOT_DIR:-$DEFAULT_ROOT_DIR}\"" in text
    assert 'PYTHON_BIN="$(resolve_python)"' in text
    assert "\n  python " not in text
    assert "\n          python " not in text


def test_ui_validation_tools_help_exits_cleanly_without_backend() -> None:
    for rel_path in [
        "tools/check_ui_endpoints.py",
        "tools/run_ui_contract_tests.py",
        "tools/run_ui_negative_tests.py",
        "tools/run_ui_param_sweep.py",
    ]:
        result = subprocess.run(
            [str(REPO_ROOT / ".venv-macos" / "bin" / "python"), str(REPO_ROOT / rel_path), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, rel_path
        assert "usage:" in result.stdout
        assert "Traceback" not in result.stderr
        assert "Backend not reachable" not in result.stdout
