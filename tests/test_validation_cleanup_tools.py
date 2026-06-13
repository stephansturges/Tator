from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import urllib.error
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


class _FakeHttpResponse:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._payload


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


def test_ui_e2e_dataset_cleanup_retries_annotation_lock(monkeypatch) -> None:
    for name, path in [
        ("tests", REPO_ROOT / "tests"),
        ("tests.ui", REPO_ROOT / "tests" / "ui"),
        ("tests.ui.e2e", REPO_ROOT / "tests" / "ui" / "e2e"),
        ("tests.ui.e2e.helpers", REPO_ROOT / "tests" / "ui" / "e2e" / "helpers"),
    ]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, package)
    _load_tool_module(
        "tests.ui.e2e.helpers.env",
        "tests/ui/e2e/helpers/env.py",
    )
    e2e_api = _load_tool_module(
        "tests.ui.e2e.helpers.api",
        "tests/ui/e2e/helpers/api.py",
    )

    calls: list[str] = []

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url,
                409,
                "Conflict",
                hdrs=None,
                fp=io.BytesIO(b'{"detail":"dataset_delete_blocked_annotation_lock"}'),
            )
        return _FakeHttpResponse(200, b'{"status":"deleted"}')

    monkeypatch.setattr(e2e_api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(e2e_api.time, "sleep", lambda _delay: None)
    monkeypatch.setenv("UI_API_ROOT", "http://example.test")

    e2e_api.delete_dataset_if_exists("pw_linked_unit", attempts=2, delay_s=0.0)

    assert calls == [
        "http://example.test/datasets/pw_linked_unit",
        "http://example.test/datasets/pw_linked_unit",
    ]


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


def test_gpu_validation_request_events_include_inflight_request(tmp_path: Path) -> None:
    tool = _load_tool_module("run_gpu_validation_suite_request_events", "tools/run_gpu_validation_suite.py")
    suite = tool.GpuValidationSuite(
        repo_root=tmp_path,
        base_url="http://127.0.0.1:8000",
        timeout_s=5,
        run_id="unit",
        cleanup=True,
    )

    class FakeSession:
        headers: dict[str, str] = {}

        def request(self, **_kwargs):
            raise TimeoutError("unit timeout")

    suite.session = FakeSession()

    status, body, elapsed, error = suite._request("GET", "/unit/hang", timeout=7)

    assert status is None
    assert body is None
    assert elapsed >= 0
    assert "unit timeout" in str(error)
    events = [
        json.loads(line)
        for line in suite.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["type"] == "request_start"
    assert events[0]["path"] == "/unit/hang"
    assert events[0]["timeout_s"] == 7
    assert events[-1]["type"] == "request_end"
    assert events[-1]["error"].startswith("request_failed:")


def test_gpu_validation_bootstrap_writes_cleanup_manifest_after_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool_module("run_gpu_validation_suite_bootstrap_seed", "tools/run_gpu_validation_suite.py")
    suite = tool.GpuValidationSuite(
        repo_root=tmp_path,
        base_url="http://127.0.0.1:8000",
        timeout_s=5,
        run_id="unit",
        cleanup=True,
    )
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"unit image")
    dataset_id = f"{suite.run_id}_ds"

    monkeypatch.setattr(suite, "_load_fixture_image", lambda _explicit: image_path)
    monkeypatch.setattr(suite, "_read_labelmap", lambda: ["object"])
    monkeypatch.setattr(suite, "_load_glossary", lambda: "{}")
    monkeypatch.setattr(suite, "_create_test_dataset_zip", lambda image_path: tmp_path / "dataset.zip")
    monkeypatch.setattr(suite, "_upload_test_dataset", lambda _zip_path: dataset_id)
    monkeypatch.setattr(suite, "_fetch_first_classifier_path", lambda: None)
    monkeypatch.setattr(suite, "_fetch_first_recipe", lambda: None)

    def fake_fetch_categories(_dataset_id: str):
        seed = json.loads(suite.cleanup_manifest_path.read_text(encoding="utf-8"))
        assert seed["status"] == "bootstrap_dataset_uploaded"
        assert seed["dataset_ids"] == [dataset_id]
        return [{"id": 1, "name": "object"}]

    monkeypatch.setattr(suite, "_fetch_dataset_categories", fake_fetch_categories)

    def fake_request(_method, path, **_kwargs):
        assert path in {"/clip/active_model", "/yolo/active", "/rfdetr/active"}
        return 200, {}, 0.0, None

    monkeypatch.setattr(suite, "_request", fake_request)

    ctx = suite.bootstrap(fixture_image=None)

    assert ctx.dataset_id == dataset_id
    assert ctx.created_dataset_ids == [dataset_id]
    final_manifest = json.loads(suite.cleanup_manifest_path.read_text(encoding="utf-8"))
    assert final_manifest["status"] == "bootstrap_complete"
    assert final_manifest["dataset_ids"] == [dataset_id]


def test_gpu_validation_job_start_persists_cleanup_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool_module("run_gpu_validation_suite_job_manifest", "tools/run_gpu_validation_suite.py")
    suite = tool.GpuValidationSuite(
        repo_root=tmp_path,
        base_url="http://127.0.0.1:8000",
        timeout_s=5,
        run_id="unit",
        cleanup=True,
    )
    dataset_id = f"{suite.run_id}_ds"
    manifest = {
        "run_id": suite.run_id,
        "artifact_root": str(suite.artifact_root),
        "upload_ns_root": str(suite.upload_ns_root),
        "dataset_ids": [dataset_id],
        "job_ids": {},
        "paths": [],
    }
    suite.ctx = tool.RunContext(
        run_id=suite.run_id,
        repo_root=tmp_path,
        base_url=suite.base_url,
        artifact_root=suite.artifact_root,
        upload_ns_root=suite.upload_ns_root,
        sample_image_path=tmp_path / "fixture.png",
        sample_image_b64="",
        labelmap=["object"],
        glossary="{}",
        dataset_id=dataset_id,
        dataset_classes=[{"id": 1, "name": "object"}],
        classifier_path=None,
        clip_active_payload={},
        yolo_active_payload={},
        rfdetr_active_payload={},
        cleanup_manifest=manifest,
    )
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, **_kwargs):
        calls.append((method, path))
        if method == "POST":
            return 200, {"job_id": "job_123"}, 0.0, None
        return 200, {"status": "completed"}, 0.0, None

    monkeypatch.setattr(suite, "_request", fake_request)

    result = suite._start_job_and_validate(
        check_id="JOB-UNIT",
        phase="jobs",
        start_path="/calibration/jobs",
        start_payload={"dataset_id": dataset_id},
        get_path_tmpl="/calibration/jobs/{job_id}",
        cancel_path_tmpl="/calibration/jobs/{job_id}/cancel",
        timeout_s=10,
        poll_s=0.25,
    )

    assert result.ok
    assert calls[0] == ("POST", "/calibration/jobs")
    persisted = json.loads(suite.cleanup_manifest_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "job_started:JOB-UNIT"
    assert persisted["job_ids"] == {"/calibration/jobs": ["job_123"]}
    assert persisted["job_cancel_paths"] == {
        "/calibration/jobs": "/calibration/jobs/{job_id}/cancel"
    }


def test_gpu_validation_cleanup_cancels_jobs_before_dataset_delete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool_module("run_gpu_validation_suite_cleanup_cancel", "tools/run_gpu_validation_suite.py")
    suite = tool.GpuValidationSuite(
        repo_root=tmp_path,
        base_url="http://127.0.0.1:8000",
        timeout_s=5,
        run_id="unit",
        cleanup=True,
    )
    dataset_id = f"{suite.run_id}_ds"
    manifest = {
        "run_id": suite.run_id,
        "artifact_root": str(suite.artifact_root),
        "upload_ns_root": str(suite.upload_ns_root),
        "dataset_ids": [dataset_id],
        "job_ids": {"/calibration/jobs": ["job_123"]},
        "paths": [],
    }
    suite.ctx = tool.RunContext(
        run_id=suite.run_id,
        repo_root=tmp_path,
        base_url=suite.base_url,
        artifact_root=suite.artifact_root,
        upload_ns_root=suite.upload_ns_root,
        sample_image_path=tmp_path / "fixture.png",
        sample_image_b64="",
        labelmap=["object"],
        glossary="{}",
        dataset_id=dataset_id,
        dataset_classes=[{"id": 1, "name": "object"}],
        classifier_path=None,
        clip_active_payload={},
        yolo_active_payload={},
        rfdetr_active_payload={},
        created_dataset_ids=[dataset_id],
        cleanup_manifest=manifest,
    )
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, **_kwargs):
        calls.append((method, path))
        if path == "/calibration/jobs/job_123/cancel":
            return 200, {"status": "cancelled"}, 0.0, None
        if path == f"/datasets/{dataset_id}":
            return 200, {"status": "deleted"}, 0.0, None
        if path == "/datasets":
            return 200, [], 0.0, None
        return 200, [], 0.0, None

    monkeypatch.setattr(suite, "_request", fake_request)

    summary = suite.cleanup()

    assert summary["cleanup_enabled"] is True
    assert calls.index(("POST", "/calibration/jobs/job_123/cancel")) < calls.index(
        ("DELETE", f"/datasets/{dataset_id}")
    )
    events = [
        json.loads(line)
        for line in suite.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event.get("type") == "cleanup_job_cancel"
        and event.get("job_id") == "job_123"
        and event.get("ok") is True
        for event in events
    )


def test_ui_param_sweep_import_is_side_effect_free(monkeypatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("run_ui_param_sweep must not call the backend during import")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    tool = _load_tool_module("run_ui_param_sweep", "tools/run_ui_param_sweep.py")

    assert callable(tool.run_sweep)
    assert callable(tool.main)


def test_ui_smoke_import_is_side_effect_free(monkeypatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("run_ui_smoke must not call the backend during import")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    tool = _load_tool_module("run_ui_smoke_import", "tools/run_ui_smoke.py")

    assert callable(tool.parse_args)
    assert callable(tool.main)


def test_ui_smoke_base_url_defaults_to_env(monkeypatch) -> None:
    tool = _load_tool_module("run_ui_smoke_args", "tools/run_ui_smoke.py")
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:9999")

    assert tool.parse_args([]).base_url == "http://127.0.0.1:9999"
    assert tool.parse_args(["http://example.test"]).base_url == "http://example.test"
    assert (
        tool.parse_args(
            ["http://example.test", "--base-url", "http://flag.test"]
        ).base_url
        == "http://flag.test"
    )


def test_ui_concurrency_smoke_import_is_side_effect_free(monkeypatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError(
            "run_ui_concurrency_smoke must not call the backend during import"
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    tool = _load_tool_module(
        "run_ui_concurrency_smoke_import",
        "tools/run_ui_concurrency_smoke.py",
    )

    assert callable(tool.parse_args)
    assert callable(tool.run_smoke)
    assert callable(tool.main)


def test_ui_concurrency_smoke_base_url_defaults_to_env(monkeypatch) -> None:
    tool = _load_tool_module(
        "run_ui_concurrency_smoke_args",
        "tools/run_ui_concurrency_smoke.py",
    )
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:9999")

    assert tool.parse_args([]).base_url == "http://127.0.0.1:9999"
    assert tool.parse_args(["http://example.test"]).base_url == "http://example.test"
    assert (
        tool.parse_args(
            ["http://example.test", "--base-url", "http://flag.test"]
        ).base_url
        == "http://flag.test"
    )


def test_ui_data_ops_import_is_side_effect_free(monkeypatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("run_ui_data_ops_tests must not call the backend during import")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    tool = _load_tool_module("run_ui_data_ops_import", "tools/run_ui_data_ops_tests.py")

    assert callable(tool.parse_args)
    assert callable(tool.main)


def test_ui_data_ops_base_url_defaults_to_env(monkeypatch) -> None:
    tool = _load_tool_module("run_ui_data_ops_args", "tools/run_ui_data_ops_tests.py")
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:9999")

    assert tool.parse_args([]).base_url == "http://127.0.0.1:9999"
    assert tool.parse_args(["http://example.test"]).base_url == "http://example.test"
    assert (
        tool.parse_args(
            ["http://example.test", "--base-url", "http://flag.test"]
        ).base_url
        == "http://flag.test"
    )


def test_watch_calibration_job_is_self_contained_and_configurable() -> None:
    path = REPO_ROOT / "tools/watch_calibration_job.sh"
    text = path.read_text(encoding="utf-8")

    assert "/tmp/print_job.py" not in text
    assert "python3 -m json.tool" in text
    assert "--base-url" in text
    assert 'BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"' in text

    syntax = subprocess.run(
        ["bash", "-n", str(path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        ["bash", str(path), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--base-url URL" in help_result.stdout


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


def test_fuzz_tier1_timeout_requests_qwen_cancel(monkeypatch) -> None:
    tool = _load_tool_module("fuzz_tier1_timeout_test", "tools/fuzz_tier1.py")
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"cancelled":true}'

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, timeout))
        if req.full_url.endswith("/qwen/cancel?force=false"):
            return Response()
        raise TimeoutError("simulated slow qwen request")

    monkeypatch.setattr(tool.urllib.request, "urlopen", fake_urlopen)

    try:
        tool._post(
            "http://127.0.0.1:8000/qwen/prepass",
            {"prepass_only": True},
            timeout=0.01,
            cancel_url="http://127.0.0.1:8000/qwen/cancel?force=false",
        )
    except RuntimeError as exc:
        assert str(exc).startswith("timeout_after_0.01s:")
    else:
        raise AssertionError("timed out Qwen fuzz request should fail")

    assert calls == [
        ("http://127.0.0.1:8000/qwen/prepass", 0.01),
        ("http://127.0.0.1:8000/qwen/cancel?force=false", 10),
    ]


def test_fuzz_tier1_wrapped_timeout_is_reported_as_timeout(monkeypatch) -> None:
    tool = _load_tool_module("fuzz_tier1_wrapped_timeout_test", "tools/fuzz_tier1.py")
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if req.full_url.endswith("/qwen/cancel?force=false"):
            return Response()
        raise tool.URLError(tool.socket.timeout("wrapped slow request"))

    monkeypatch.setattr(tool.urllib.request, "urlopen", fake_urlopen)

    try:
        tool._post(
            "http://127.0.0.1:8000/qwen/caption",
            {},
            timeout=0.02,
            cancel_url="http://127.0.0.1:8000/qwen/cancel?force=false",
        )
    except RuntimeError as exc:
        assert str(exc).startswith("timeout_after_0.02s:")
    else:
        raise AssertionError("wrapped timeout should fail as timeout")

    assert calls == [
        "http://127.0.0.1:8000/qwen/caption",
        "http://127.0.0.1:8000/qwen/cancel?force=false",
    ]


def test_fuzz_tier1_failure_writes_structured_summary(tmp_path: Path) -> None:
    tool = _load_tool_module("fuzz_tier1_summary_test", "tools/fuzz_tier1.py")
    out = tmp_path / "tier1.json"
    summary = {"skip_gpu": False, "steps": [{"name": "prepass_base", "result": {"ok": True}}]}

    code = tool._record_failure(summary, str(out), "prepass_windowed", RuntimeError("timeout_after_60s"))

    assert code == 1
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["steps"][-1] == {
        "name": "prepass_windowed",
        "failed": True,
        "error": "timeout_after_60s",
    }


def test_run_fuzz_fast_exposes_tier1_request_timeout() -> None:
    text = (REPO_ROOT / "tools/run_fuzz_fast.sh").read_text(encoding="utf-8")

    assert 'REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"' in text
    assert '--request-timeout "$REQUEST_TIMEOUT"' in text


def test_ui_validation_tools_help_exits_cleanly_without_backend() -> None:
    for rel_path in [
        "tools/check_ui_endpoints.py",
        "tools/derive_context_feature_variants.py",
        "tools/detect_missclassifications.py",
        "tools/fuzz_tier0.py",
        "tools/fuzz_tier1.py",
        "tools/label_candidates_iou90.py",
        "tools/run_class_split_qwen_review_benchmark.py",
        "tools/run_context_feature_ablation.py",
        "tools/run_ui_contract_tests.py",
        "tools/run_ui_concurrency_smoke.py",
        "tools/run_ui_data_ops_tests.py",
        "tools/run_ui_negative_tests.py",
        "tools/run_ui_param_sweep.py",
        "tools/run_ui_smoke.py",
        "tools/watch_yolo_train_and_activate.py",
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
        assert "Loading CLIP model" not in result.stdout
        assert "Loading CLIP model" not in result.stderr


def test_unused_def_scanner_counts_first_party_models_package_uses(
    tmp_path: Path,
) -> None:
    tool = _load_tool_module("scan_unused_defs_test", "tools/scan_unused_defs.py")
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "compat.py").write_text(
        "def root_validator_compat(*args, **kwargs):\n"
        "    return lambda fn: fn\n",
        encoding="utf-8",
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "schemas.py").write_text(
        "from utils.compat import root_validator_compat\n\n"
        "@root_validator_compat(skip_on_failure=True)\n"
        "def validate_payload(cls, values):\n"
        "    return values\n",
        encoding="utf-8",
    )

    uses = tool.collect_uses(tmp_path)

    assert uses["root_validator_compat"] >= 1
