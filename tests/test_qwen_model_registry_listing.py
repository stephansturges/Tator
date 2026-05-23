from __future__ import annotations

import json
from pathlib import Path

import localinferenceapi as api


def _write_meta(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / api.QWEN_METADATA_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _write_transformers_adapter(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"adapter")


def test_list_qwen_model_entries_skips_incomplete_and_dedupes(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "qwen_runs"
    runs = root / "runs"
    runs.mkdir(parents=True)

    # Valid run with explicit checkpoint path.
    run_a = runs / "run_a"
    ckpt_a = run_a / "latest"
    _write_transformers_adapter(ckpt_a)
    _write_meta(
        run_a,
        {
            "id": "model_a",
            "label": "Model A",
            "latest_checkpoint": str(ckpt_a),
            "created_at": 200.0,
        },
    )

    # Duplicate id should be dropped in favor of newest entry.
    run_dup = runs / "run_dup"
    ckpt_dup = run_dup / "latest"
    _write_transformers_adapter(ckpt_dup)
    _write_meta(
        run_dup,
        {
            "id": "model_a",
            "label": "Model A older",
            "latest_checkpoint": str(ckpt_dup),
            "created_at": 100.0,
        },
    )

    # Incomplete run should be ignored.
    run_bad = runs / "run_bad"
    _write_meta(
        run_bad,
        {
            "id": "model_bad",
            "label": "Broken",
            "latest_checkpoint": str(run_bad / "missing_checkpoint"),
            "created_at": 300.0,
        },
    )

    monkeypatch.setattr(api, "QWEN_JOB_ROOT", root)

    entries = api._list_qwen_model_entries()
    assert len(entries) == 1
    assert entries[0]["id"] == "model_a"
    assert entries[0]["label"] == "Model A"
    assert entries[0]["path"] == str(ckpt_a.resolve())


def test_list_qwen_model_entries_uses_latest_dir_without_metadata(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "qwen_runs"
    run = root / "runs" / "run_no_meta"
    latest = run / "latest"
    _write_transformers_adapter(latest)

    monkeypatch.setattr(api, "QWEN_JOB_ROOT", root)

    entries = api._list_qwen_model_entries()
    assert len(entries) == 1
    assert entries[0]["id"] == "run_no_meta"
    assert entries[0]["path"] == str(latest.resolve())


def test_list_qwen_model_entries_skips_mlx_adapters_without_loader_config(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "qwen_runs"
    runs = root / "runs"
    good = runs / "good_mlx"
    good_latest = good / "latest"
    good_latest.mkdir(parents=True)
    (good_latest / "adapters.safetensors").write_bytes(b"stub")
    (good_latest / "adapter_config.json").write_text(
        '{"rank": 8, "alpha": 2.0, "dropout": 0.05}',
        encoding="utf-8",
    )
    _write_meta(
        good,
        {
            "id": "good_mlx",
            "latest_checkpoint": str(good_latest),
            "runtime_platform": api.QWEN_PLATFORM_MLX,
            "created_at": 200.0,
        },
    )

    broken = runs / "broken_mlx"
    broken_latest = broken / "latest"
    broken_latest.mkdir(parents=True)
    (broken_latest / "adapters.safetensors").write_bytes(b"stub")
    _write_meta(
        broken,
        {
            "id": "broken_mlx",
            "latest_checkpoint": str(broken_latest),
            "runtime_platform": api.QWEN_PLATFORM_MLX,
            "created_at": 300.0,
        },
    )
    broken_inferred = runs / "broken_inferred_mlx"
    broken_inferred_latest = broken_inferred / "latest"
    broken_inferred_latest.mkdir(parents=True)
    (broken_inferred_latest / "adapters.safetensors").write_bytes(b"stub")
    _write_meta(
        broken_inferred,
        {
            "id": "broken_inferred_mlx",
            "model_id": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            "latest_checkpoint": str(broken_inferred_latest),
            "created_at": 250.0,
        },
    )

    monkeypatch.setattr(api, "QWEN_JOB_ROOT", root)

    entries = api._list_qwen_model_entries()
    assert [entry["id"] for entry in entries] == ["good_mlx"]
