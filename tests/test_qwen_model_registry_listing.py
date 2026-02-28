from __future__ import annotations

import json
from pathlib import Path

import localinferenceapi as api


def _write_meta(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / api.QWEN_METADATA_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def test_list_qwen_model_entries_skips_incomplete_and_dedupes(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "qwen_runs"
    runs = root / "runs"
    runs.mkdir(parents=True)

    # Valid run with explicit checkpoint path.
    run_a = runs / "run_a"
    ckpt_a = run_a / "latest"
    ckpt_a.mkdir(parents=True)
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
    ckpt_dup.mkdir(parents=True)
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
    latest.mkdir(parents=True)

    monkeypatch.setattr(api, "QWEN_JOB_ROOT", root)

    entries = api._list_qwen_model_entries()
    assert len(entries) == 1
    assert entries[0]["id"] == "run_no_meta"
    assert entries[0]["path"] == str(latest.resolve())
