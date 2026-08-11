from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from PIL import Image

from tools import preflight_qwen_caption_soak as preflight


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 16), color=(10, 20, 30)).save(path)


def _dataset(tmp_path: Path, names: list[str]) -> Path:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    for name in names:
        _write_image(dataset / "images" / f"{name}.jpg")
    return dataset


def _args(dataset: Path, output_dir: Path, **overrides):
    data = {
        "dataset_root": dataset,
        "cases_json": None,
        "request_json": None,
        "output_dir": output_dir,
        "all_images": True,
        "caption_mode": "full",
        "sample_size": 0,
        "sample_seed": 13,
        "case": [],
        "limit": 0,
        "resume": True,
        "save_dataset_text_labels": False,
        "attempts": 2,
        "max_artifact_log_bytes": 1024,
        "min_free_gb": 0,
        "disk_safety_factor": 1.0,
        "max_heartbeat_age": 60,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _patch_disk(monkeypatch, free: int = 10_000_000_000) -> None:
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=free * 2, used=free, free=free),
    )


def _checks(report):
    return {check["name"]: check for check in report["checks"]}


def _write_cached_model(tmp_path: Path, monkeypatch, model_id: str) -> Path:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    repo = tmp_path / "hf" / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / "abcdef"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text("abcdef")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def test_preflight_ok_for_fresh_dataset_soak(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a", "b"])
    _patch_disk(monkeypatch)

    report = preflight.preflight_soak(_args(dataset, tmp_path / "run"))

    assert report["status"] == "ok"
    assert report["resume"]["remaining_cases"] == 2
    assert report["disk"]["bounded"] is True
    assert _checks(report)["case_selection"]["status"] == "ok"
    assert _checks(report)["disk_budget"]["status"] == "ok"


def test_preflight_case_selection_uses_stress_sample_strategy(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Boat\nBuilding\nPerson\nVehicle\n")
    label_dir = dataset / "train" / "labels"
    for stem in ["empty", "sparse", "diverse", "dominant", "dense"]:
        _write_image(dataset / "train" / "images" / f"{stem}.jpg")
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "empty.txt").write_text("")
    (label_dir / "sparse.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (label_dir / "diverse.txt").write_text(
        "".join(f"{index % 4} 0.5 0.5 0.1 0.1\n" for index in range(20))
    )
    (label_dir / "dominant.txt").write_text("0 0.5 0.5 0.1 0.1\n" * 40)
    (label_dir / "dense.txt").write_text(("1 0.5 0.5 0.1 0.1\n" * 45) + ("0 0.5 0.5 0.1 0.1\n" * 15))

    cases, meta = preflight._load_requested_cases(
        _args(dataset, tmp_path / "run", sample_size=2, sample_seed=123)
    )

    assert [preflight.runner.case_key(case) for case in cases] == [
        "image:dense:full",
        "image:diverse:full",
    ]
    assert meta["sample_selection"]["strategy"] == preflight.runner.SAMPLE_STRATEGY_STRESS_PLUS_RANDOM
    assert meta["sample_selection"]["source_cases"] == 5
    assert meta["sample_selection"]["selected_cases"] == 2
    assert meta["sample_selection"]["stress_case_keys"] == [
        "image:dense:full",
        "image:diverse:full",
    ]


def test_preflight_errors_when_selected_model_cache_is_missing(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    _patch_disk(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    report = preflight.preflight_soak(
        _args(
            dataset,
            tmp_path / "run",
            model_id="example/missing-model",
            model_variant="auto",
            refinement_model_id="same",
            fallback_model_id="none",
            loop_recovery="off",
            allow_model_download=False,
            preview_only=False,
        )
    )

    assert report["status"] == "error"
    assert _checks(report)["model_cache"]["status"] == "error"
    assert "download or choose a local model" in _checks(report)["model_cache"]["detail"]


def test_preflight_allows_intentional_model_download_as_warning(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    _patch_disk(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    report = preflight.preflight_soak(
        _args(
            dataset,
            tmp_path / "run",
            model_id="example/missing-model",
            model_variant="auto",
            refinement_model_id="same",
            fallback_model_id="none",
            loop_recovery="off",
            allow_model_download=True,
            preview_only=False,
        )
    )

    assert report["status"] == "warn"
    assert _checks(report)["model_cache"]["status"] == "warn"
    assert report["model_cache"]["models"][0]["needs_download"] is True


def test_preflight_accepts_cached_caption_refinement_and_fallback_models(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    _patch_disk(monkeypatch)
    _write_cached_model(tmp_path, monkeypatch, "example/caption-model")
    _write_cached_model(tmp_path, monkeypatch, "example/refinement-model")
    _write_cached_model(tmp_path, monkeypatch, "example/fallback-model")

    report = preflight.preflight_soak(
        _args(
            dataset,
            tmp_path / "run",
            model_id="example/caption-model",
            model_variant="auto",
            refinement_model_id="example/refinement-model",
            fallback_model_id="example/fallback-model",
            loop_recovery="safe_retry_fallback",
            allow_model_download=False,
            preview_only=False,
        )
    )

    assert report["status"] == "ok"
    assert _checks(report)["model_cache"]["status"] == "ok"
    assert {item["role"] for item in report["model_cache"]["models"]} == {
        "caption",
        "refinement",
        "fallback",
    }


def test_preflight_applies_request_json_model_overrides(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    _patch_disk(monkeypatch)
    _write_cached_model(tmp_path, monkeypatch, "example/template-model")
    request_json = tmp_path / "request.json"
    request_json.write_text(
        json.dumps({
            "model_id": "example/template-model",
            "model_variant": "auto",
            "refinement_model_id": "same",
            "caption_fallback_model_id": "none",
            "caption_loop_recovery_mode": "off",
            "image_name": "must-not-apply.jpg",
            "label_hints": [],
        })
    )

    report = preflight.preflight_soak(
        _args(
            dataset,
            tmp_path / "run",
            request_json=request_json,
            model_id="example/missing-cli-model",
            model_variant="auto",
            refinement_model_id="same",
            fallback_model_id="none",
            loop_recovery="off",
            allow_model_download=False,
            preview_only=False,
        )
    )

    assert report["status"] == "ok"
    assert _checks(report)["request_template"]["status"] == "ok"
    assert report["request_template"]["ignored_image_keys"] == ["image_name", "label_hints"]
    assert _checks(report)["model_cache"]["status"] == "ok"
    assert [(item["role"], item["model_id"]) for item in report["model_cache"]["models"]] == [
        ("caption", "example/template-model"),
        ("refinement", "example/template-model"),
    ]
    assert report["model_cache"]["request_model_overrides"]["model_id"] == "example/template-model"


def test_preflight_errors_when_request_json_is_missing(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    _patch_disk(monkeypatch)

    report = preflight.preflight_soak(
        _args(
            dataset,
            tmp_path / "run",
            request_json=tmp_path / "missing-request.json",
            model_id="example/model",
            model_variant="auto",
            refinement_model_id="same",
            fallback_model_id="none",
            loop_recovery="off",
            allow_model_download=True,
            preview_only=False,
        )
    )

    assert report["status"] == "error"
    assert _checks(report)["request_template"]["status"] == "error"
    assert "request template not found" in _checks(report)["request_template"]["detail"]


def test_preflight_preview_only_skips_model_cache_requirement(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    _patch_disk(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    report = preflight.preflight_soak(
        _args(
            dataset,
            tmp_path / "run",
            model_id="example/missing-model",
            model_variant="auto",
            refinement_model_id="same",
            fallback_model_id="none",
            loop_recovery="off",
            allow_model_download=False,
            preview_only=True,
        )
    )

    assert report["status"] == "ok"
    assert _checks(report)["model_cache"]["status"] == "ok"
    assert report["model_cache"]["preview_only"] is True


def test_preflight_errors_on_live_runner_lock(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _patch_disk(monkeypatch)
    (output_dir / preflight.runner.RUNNER_LOCK_NAME).write_text(
        json.dumps({
            "runner_id": "live",
            "pid": os.getpid(),
            "heartbeat_epoch": preflight.time.time(),
            "runner_capabilities": [preflight.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART],
        })
    )
    probe_calls: list[Path] = []

    def fake_probe(path: Path, *, label: str):
        probe_calls.append(path)
        return {"status": "ok", "detail": f"{label} is writable", "path": str(path)}

    monkeypatch.setattr(preflight, "_probe_directory_writable", fake_probe)

    report = preflight.preflight_soak(_args(dataset, output_dir))

    assert report["status"] == "error"
    assert _checks(report)["runner_lock"]["status"] == "error"
    assert "live runner" in _checks(report)["runner_lock"]["detail"]
    assert _checks(report)["runner_lock"]["runner_supports_graceful_restart"] is True
    assert preflight.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART in _checks(report)["runner_lock"]["runner_capabilities"]
    assert _checks(report)["artifact_write"]["status"] == "ok"
    assert _checks(report)["artifact_write"]["skipped"] is True
    assert output_dir not in probe_calls


def test_preflight_warns_on_invalid_runner_lock_and_keeps_write_probe(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _patch_disk(monkeypatch)
    (output_dir / preflight.runner.RUNNER_LOCK_NAME).write_text("{not-json")
    probe_calls: list[Path] = []

    def fake_probe(path: Path, *, label: str):
        probe_calls.append(path)
        return {"status": "ok", "detail": f"{label} is writable", "path": str(path)}

    monkeypatch.setattr(preflight, "_probe_directory_writable", fake_probe)

    report = preflight.preflight_soak(_args(dataset, output_dir))

    assert report["status"] == "warn"
    assert _checks(report)["runner_lock"]["status"] == "warn"
    assert "invalid runner lock" in _checks(report)["runner_lock"]["detail"]
    assert "runner can remove it before resume" in _checks(report)["runner_lock"]["detail"]
    assert _checks(report)["artifact_write"]["status"] == "ok"
    assert "skipped" not in _checks(report)["artifact_write"]
    assert output_dir in probe_calls


def test_preflight_errors_on_resume_manifest_case_mismatch(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _patch_disk(monkeypatch)
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "cases": [
                {
                    "name": "image_000001",
                    "stem": "different",
                    "caption_mode": "full",
                }
            ]
        })
    )

    report = preflight.preflight_soak(_args(dataset, output_dir, resume=True))

    assert report["status"] == "error"
    assert _checks(report)["resume_manifest"]["status"] == "error"
    assert "does not match" in _checks(report)["resume_manifest"]["detail"]


def test_preflight_errors_on_resume_manifest_run_settings_mismatch(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _patch_disk(monkeypatch)
    original_args = _args(dataset, output_dir, resume=True, preview_only=True, prompt="Describe.")
    cases = preflight.runner.select_all_image_cases(
        preflight.runner.discover_items(dataset),
        caption_mode="full",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "cases": cases,
            "run_settings": preflight.runner.run_settings_payload(original_args, request_template={}),
        })
    )

    report = preflight.preflight_soak(
        _args(dataset, output_dir, resume=True, preview_only=True, prompt="Different prompt.")
    )

    assert report["status"] == "error"
    assert _checks(report)["resume_manifest"]["status"] == "ok"
    assert _checks(report)["resume_settings"]["status"] == "error"
    assert "do not match" in _checks(report)["resume_settings"]["detail"]
    assert _checks(report)["resume_settings"]["existing_fingerprint"]
    assert _checks(report)["resume_settings"]["requested_fingerprint"]


def test_preflight_resume_settings_allow_request_json_path_change_with_same_template(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    old_metadata_dir = tmp_path / "job_old"
    new_metadata_dir = tmp_path / "job_new"
    old_metadata_dir.mkdir()
    new_metadata_dir.mkdir()
    old_request = old_metadata_dir / "request_fields.json"
    new_request = new_metadata_dir / "request_fields.json"
    request_template = {"user_prompt": "Describe consistently.", "image_name": "ignored.jpg"}
    old_request.write_text(json.dumps(request_template))
    new_request.write_text(json.dumps(request_template))
    _patch_disk(monkeypatch)
    original_args = _args(
        dataset,
        output_dir,
        resume=True,
        preview_only=True,
        request_json=old_request,
    )
    cases = preflight.runner.select_all_image_cases(
        preflight.runner.discover_items(dataset),
        caption_mode="full",
    )
    case_id = preflight.runner.case_key(cases[0])
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "cases": cases,
            "run_settings": preflight.runner.run_settings_payload(original_args),
        })
    )
    (output_dir / "results.jsonl").write_text(
        json.dumps({
            "case_id": case_id,
            "final_status": "preview_only",
            "status": "preview_only",
            "quality_failures": [],
        })
        + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"total_cases": 1, "totals": {"preview_only": 1}})
    )
    (output_dir / "heartbeat.json").write_text(
        json.dumps({"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()})
    )

    report = preflight.preflight_soak(
        _args(
            dataset,
            output_dir,
            resume=True,
            preview_only=True,
            request_json=new_request,
        )
    )

    assert report["status"] == "ok"
    assert _checks(report)["resume_manifest"]["status"] == "ok"
    assert _checks(report)["resume_settings"]["status"] == "ok"
    assert (
        _checks(report)["resume_settings"]["existing_fingerprint"]
        == _checks(report)["resume_settings"]["requested_fingerprint"]
    )


def test_preflight_resume_settings_allow_cases_json_path_change_with_same_cases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    old_metadata_dir = tmp_path / "job_old"
    new_metadata_dir = tmp_path / "job_new"
    old_metadata_dir.mkdir()
    new_metadata_dir.mkdir()
    _patch_disk(monkeypatch)
    cases = preflight.runner.select_all_image_cases(
        preflight.runner.discover_items(dataset),
        caption_mode="full",
    )
    old_cases_json = old_metadata_dir / "cases.json"
    new_cases_json = new_metadata_dir / "cases.json"
    old_cases_json.write_text(json.dumps(cases))
    new_cases_json.write_text(json.dumps(cases))
    case_id = preflight.runner.case_key(cases[0])
    original_args = _args(
        dataset,
        output_dir,
        cases_json=old_cases_json,
        resume=True,
        preview_only=True,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "cases": cases,
            "run_settings": preflight.runner.run_settings_payload(original_args),
        })
    )
    (output_dir / "results.jsonl").write_text(
        json.dumps({
            "case_id": case_id,
            "final_status": "preview_only",
            "status": "preview_only",
            "quality_failures": [],
        })
        + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"total_cases": 1, "totals": {"preview_only": 1}})
    )
    (output_dir / "heartbeat.json").write_text(
        json.dumps({"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()})
    )

    report = preflight.preflight_soak(
        _args(
            dataset,
            output_dir,
            cases_json=new_cases_json,
            resume=True,
            preview_only=True,
        )
    )

    assert report["status"] == "ok"
    assert _checks(report)["resume_manifest"]["status"] == "ok"
    assert _checks(report)["resume_settings"]["status"] == "ok"
    assert (
        _checks(report)["resume_settings"]["existing_fingerprint"]
        == _checks(report)["resume_settings"]["requested_fingerprint"]
    )


def test_preflight_errors_on_invalid_resume_results_jsonl(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _patch_disk(monkeypatch)
    (output_dir / "results.jsonl").write_text(
        json.dumps({
            "case_id": "image:a:full",
            "final_status": "ok",
            "status": "ok",
            "quality_failures": [],
        })
        + "\n"
        + "{not valid json}\n"
    )

    report = preflight.preflight_soak(_args(dataset, output_dir, resume=True))

    assert report["status"] == "error"
    assert _checks(report)["resume_rows"]["status"] == "error"
    assert "1 invalid row" in _checks(report)["resume_rows"]["detail"]
    assert _checks(report)["resume_rows"]["invalid_rows"][0]["line"] == 2
    assert _checks(report)["artifact_audit"]["status"] == "error"
    assert report["resume"]["latest_rows"] == 0


def test_preflight_errors_on_invalid_resume_captions_jsonl(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _patch_disk(monkeypatch)
    (output_dir / "manifest.json").write_text(
        json.dumps({"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]})
    )
    (output_dir / "results.jsonl").write_text(
        json.dumps({
            "case_id": "image:a:full",
            "final_status": "ok",
            "status": "ok",
            "quality_failures": [],
        })
        + "\n"
    )
    (output_dir / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "caption": "A caption."})
        + "\n"
        + "{not valid json}\n"
    )
    (output_dir / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
    (output_dir / "heartbeat.json").write_text(
        json.dumps({"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()})
    )

    report = preflight.preflight_soak(_args(dataset, output_dir, resume=True))

    assert report["status"] == "error"
    assert _checks(report)["caption_rows"]["status"] == "error"
    assert "1 invalid row" in _checks(report)["caption_rows"]["detail"]
    assert _checks(report)["caption_rows"]["invalid_rows"][0]["line"] == 2
    assert _checks(report)["artifact_audit"]["status"] == "error"
    assert report["resume"]["latest_rows"] == 1


def test_preflight_warns_on_recoverable_interrupted_artifacts(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _patch_disk(monkeypatch)
    cases = preflight.runner.select_all_image_cases(
        preflight.runner.discover_items(dataset),
        caption_mode="full",
    )
    (output_dir / "manifest.json").write_text(json.dumps({"cases": cases}))
    (output_dir / "heartbeat.json").write_text(
        json.dumps({
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 1000,
        })
    )

    report = preflight.preflight_soak(
        _args(dataset, output_dir, resume=True, max_heartbeat_age=0.05)
    )

    checks = _checks(report)
    assert report["status"] == "warn"
    assert checks["resume_rows"]["status"] == "ok"
    assert checks["resume_manifest"]["status"] == "ok"
    assert checks["artifact_audit"]["status"] == "warn"
    assert checks["artifact_audit"]["audit_status"] == "error"
    assert checks["artifact_audit"]["recoverable_interrupted_state"] is True
    assert "recoverable by resume" in checks["artifact_audit"]["detail"]


def test_preflight_errors_when_artifact_output_is_not_writable(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    _patch_disk(monkeypatch)

    def fake_probe(path: Path, *, label: str):
        if path == output_dir.resolve():
            return {
                "status": "error",
                "detail": f"{label} is not writable: denied",
                "path": str(path),
                "error_type": "PermissionError",
            }
        return {"status": "ok", "detail": f"{label} is writable", "path": str(path)}

    monkeypatch.setattr(preflight, "_probe_directory_writable", fake_probe)

    report = preflight.preflight_soak(_args(dataset, output_dir))

    assert report["status"] == "error"
    assert _checks(report)["artifact_write"]["status"] == "error"
    assert "not writable" in _checks(report)["artifact_write"]["detail"]


def test_preflight_probes_dataset_text_label_dirs_when_requested(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    _patch_disk(monkeypatch)
    probed_paths: list[Path] = []

    def fake_probe(path: Path, *, label: str):
        probed_paths.append(path)
        return {"status": "ok", "detail": f"{label} is writable", "path": str(path)}

    monkeypatch.setattr(preflight, "_probe_directory_writable", fake_probe)

    report = preflight.preflight_soak(
        _args(dataset, output_dir, save_dataset_text_labels=True, preview_only=True)
    )

    assert report["status"] == "ok"
    assert _checks(report)["artifact_write"]["status"] == "ok"
    assert _checks(report)["text_label_write"]["status"] == "ok"
    assert output_dir.resolve() in probed_paths
    assert dataset / "text_labels" in probed_paths


def test_preflight_parser_exposes_caption_runner_settings() -> None:
    parser = preflight.build_parser()
    args = parser.parse_args(
        [
            "--max-boxes",
            "24",
            "--max-new-tokens",
            "1800",
            "--final-sentences",
            "5",
            "--window-size",
            "512",
            "--window-overlap",
            "0.25",
            "--mlx-max-image-side",
            "384",
            "--temperature",
            "0.35",
            "--top-p",
            "0.7",
            "--top-k",
            "10",
            "--use-sampling",
            "--save-dataset-text-labels",
            "--prompt",
            "Custom caption prompt.",
        ]
    )

    assert args.max_boxes == 24
    assert args.max_new_tokens == 1800
    assert args.final_sentences == 5
    assert args.window_size == 512
    assert args.window_overlap == 0.25
    assert args.mlx_max_image_side == 384
    assert args.temperature == 0.35
    assert args.top_p == 0.7
    assert args.top_k == 10
    assert args.use_sampling is True
    assert args.save_dataset_text_labels is True
    assert args.prompt == "Custom caption prompt."


def test_preflight_warns_when_raw_logs_are_uncapped(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    _patch_disk(monkeypatch)

    report = preflight.preflight_soak(
        _args(dataset, tmp_path / "run", max_artifact_log_bytes=0)
    )

    assert report["status"] == "warn"
    assert report["disk"]["bounded"] is False
    assert _checks(report)["disk_budget"]["status"] == "warn"
    assert "uncapped" in _checks(report)["disk_budget"]["detail"]


def test_preflight_allows_legacy_resume_rows_when_cases_json_is_authoritative(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cases_json = tmp_path / "cases.json"
    cases_json.write_text(
        json.dumps([
            {
                "name": "image_000001",
                "stem": "a",
                "caption_mode": "full",
                "image_path": str(dataset / "images" / "a.jpg"),
                "label_count": 0,
                "class_counts": {},
            }
        ])
    )
    (output_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image_000001:a:full",
                "status": "ok",
                "final_status": "ok",
                "quality_failures": [],
            }
        )
        + "\n"
    )
    (output_dir / "captions.jsonl").write_text(
        json.dumps({"case_id": "image_000001:a:full", "caption": "A caption."})
        + "\n"
    )
    _patch_disk(monkeypatch)

    report = preflight.preflight_soak(
        _args(dataset, output_dir, cases_json=cases_json, resume=True)
    )

    assert report["status"] == "warn"
    assert _checks(report)["resume_manifest"]["status"] == "warn"
    assert "using cases-json" in _checks(report)["resume_manifest"]["detail"]


def test_preflight_cli_runs_directly_with_cases_json(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, ["a"])
    cases_json = tmp_path / "cases.json"
    cases_json.write_text(
        json.dumps([
            {
                "name": "image_000001",
                "stem": "a",
                "caption_mode": "full",
                "image_path": str(dataset / "images" / "a.jpg"),
                "label_count": 0,
                "class_counts": {},
            }
        ])
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tools/preflight_qwen_caption_soak.py",
            "--dataset-root",
            str(dataset),
            "--cases-json",
            str(cases_json),
            "--output-dir",
            str(tmp_path / "run"),
            "--min-free-gb",
            "0",
            "--max-artifact-log-bytes",
            "1024",
            "--preview-only",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "ok"
    assert report["resume"]["remaining_cases"] == 1
