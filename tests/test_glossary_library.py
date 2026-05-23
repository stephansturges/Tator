from __future__ import annotations

from pathlib import Path

import pytest

import localinferenceapi as api


def test_glossary_library_rejects_blank_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "glossaries"
    root.mkdir()
    monkeypatch.setattr(api, "GLOSSARY_LIBRARY_ROOT", root)

    for name, detail in [("", "glossary_name_required"), ("...", "glossary_name_invalid")]:
        with pytest.raises(api.HTTPException) as exc_info:
            api.save_glossary_entry(name, "object")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == detail

    assert list(root.iterdir()) == []


def test_glossary_library_rejects_path_like_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "glossaries"
    root.mkdir()
    monkeypatch.setattr(api, "GLOSSARY_LIBRARY_ROOT", root)

    for name in ["../escape", r"..\escape"]:
        with pytest.raises(api.HTTPException) as exc_info:
            api.save_glossary_entry(name, "object")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "glossary_name_invalid"

    assert not (tmp_path / "escape.json").exists()
    assert list(root.iterdir()) == []


def test_glossary_library_sanitizes_stable_valid_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "glossaries"
    root.mkdir()
    monkeypatch.setattr(api, "GLOSSARY_LIBRARY_ROOT", root)

    saved = api.save_glossary_entry("Object class", "one object per line")
    loaded = api.get_glossary_entry("Object class")
    deleted = api.delete_glossary_entry("Object class")

    assert saved == {"status": "saved", "name": "Object class"}
    assert loaded == {"name": "Object class", "glossary": "one object per line"}
    assert deleted == {"status": "deleted", "name": "Object class"}
    assert not (root / "Object-class.json").exists()
