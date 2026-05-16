import os

import pytest

from localinferenceapi import (
    _copy2_if_different as _api_copy2_if_different,
    _link_or_copy_file,
    _unlink_self_referential_symlink,
)
from services.calibration_helpers import _calibration_safe_link
from services.canonical_edr_completion import _copy2_if_different as _canonical_copy2_if_different
from services.detectors import _copy2_if_different as _detector_copy2_if_different
from services.edr_packages import _copy2_if_different as _edr_copy2_if_different
from services.prepass_recipes import _copy2_if_different as _prepass_copy2_if_different


def test_link_or_copy_file_noops_when_source_is_destination(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"classifier")

    _link_or_copy_file(artifact, artifact, overwrite=True)

    assert artifact.read_bytes() == b"classifier"
    assert not artifact.is_symlink()


def test_unlink_self_referential_symlink_removes_broken_artifact(tmp_path):
    artifact = tmp_path / "model.pkl"
    os.symlink(str(artifact), artifact)

    assert artifact.is_symlink()
    assert _unlink_self_referential_symlink(artifact) is True
    assert not artifact.exists()
    assert not artifact.is_symlink()


@pytest.mark.parametrize(
    "copy_fn",
    [
        _api_copy2_if_different,
        _canonical_copy2_if_different,
        _detector_copy2_if_different,
        _edr_copy2_if_different,
        _prepass_copy2_if_different,
    ],
)
def test_copy_helpers_noop_when_source_is_destination(tmp_path, copy_fn):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"payload")

    copy_fn(artifact, artifact)

    assert artifact.read_bytes() == b"payload"
    assert not artifact.is_symlink()


@pytest.mark.parametrize(
    "copy_fn",
    [
        _api_copy2_if_different,
        _canonical_copy2_if_different,
        _detector_copy2_if_different,
        _edr_copy2_if_different,
        _prepass_copy2_if_different,
    ],
)
def test_copy_helpers_replace_self_referential_destination(tmp_path, copy_fn):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    source.write_bytes(b"payload")
    os.symlink(str(dest), dest)

    copy_fn(source, dest)

    assert dest.read_bytes() == b"payload"
    assert not dest.is_symlink()


def test_calibration_safe_link_replaces_self_referential_destination(tmp_path):
    source = tmp_path / "features.npz"
    dest = tmp_path / "cached_features.npz"
    source.write_bytes(b"features")
    os.symlink(str(dest), dest)

    _calibration_safe_link(source, dest)

    assert dest.exists()
    assert dest.resolve() == source.resolve()
