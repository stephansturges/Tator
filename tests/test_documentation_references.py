from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_tool_and_test_references_exist() -> None:
    """Keep first-layer command references tied to files that actually ship."""

    docs = ["readme.md", "tools/README.md"]
    missing: list[str] = []
    for doc in docs:
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        refs = {
            match.group(1).rstrip(".,)`:;")
            for match in re.finditer(
                r"(?<![\w./-])((?:tools|tests)/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+)",
                text,
            )
        }
        missing.extend(
            f"{doc}: {ref}"
            for ref in refs
            if "*" not in ref and not (REPO_ROOT / ref).exists()
        )

    assert sorted(missing) == []
