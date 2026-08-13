import json
from pathlib import Path

import pytest

from services.class_analysis_sessions import (
    LATEST_SESSION_FILENAME,
    SESSION_MANIFEST_FILENAME,
    SessionManifestError,
    build_session_manifest,
    latest_session,
    read_session_manifest,
    write_session_manifest,
)


def _manifest(job_id: str, *, updated_at: float = 10.0):
    return build_session_manifest(
        job_id=job_id,
        status="completed",
        summary={
            "source_mode": "active_workspace",
            "source_id": "cas_1",
            "snapshot_id": "cas_1",
            "dataset_label": "fixture",
            "image_count": 3,
            "object_count": 7,
            "quality_recipe": "thorough_quality_v1",
            "projection": "umap",
            "refinement": {"status": "completed"},
            "runtime": {"refinement": {"processed": 2, "total": 2}},
        },
        request={"deep_evidence_pass": True},
        state={
            "updated_at": updated_at,
            "result_file": "result.json",
            "result_sha256": "a" * 64,
            "result_bytes": 123,
        },
        created_at=1.0,
        updated_at=updated_at,
    )


def test_session_manifest_round_trip_and_latest_rebuild(tmp_path: Path):
    root = tmp_path / "uploads" / "class_analysis"
    root.mkdir(parents=True)
    first = write_session_manifest(root, _manifest("ca_first", updated_at=10.0))
    second = write_session_manifest(root, _manifest("ca_second", updated_at=20.0))
    assert read_session_manifest(root, "ca_first") == first
    assert latest_session(root) == second
    (root / LATEST_SESSION_FILENAME).unlink()
    assert latest_session(root) == second
    assert (root / LATEST_SESSION_FILENAME).is_file()


def test_corrupt_or_mismatched_manifest_fails_closed(tmp_path: Path):
    root = tmp_path / "uploads" / "class_analysis"
    job_dir = root / "ca_bad"
    job_dir.mkdir(parents=True)
    path = job_dir / SESSION_MANIFEST_FILENAME
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(SessionManifestError):
        read_session_manifest(root, "ca_bad")
    assert latest_session(root) is None
    path.write_text(json.dumps(_manifest("ca_other")), encoding="utf-8")
    with pytest.raises(SessionManifestError):
        read_session_manifest(root, "ca_bad")


def test_manifest_rejects_nonterminal_or_unsafe_jobs():
    with pytest.raises(SessionManifestError):
        build_session_manifest(
            job_id="../escape",
            status="completed",
            summary={},
            request={},
            state={},
        )
    with pytest.raises(SessionManifestError):
        build_session_manifest(
            job_id="ca_running",
            status="running",
            summary={},
            request={},
            state={},
        )
