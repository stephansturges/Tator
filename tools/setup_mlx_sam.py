#!/usr/bin/env python3
"""Install the official MLX SAM example and converted base model locally."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / ".cache" / "tator" / "mlx-sam"
UPSTREAM_REPO = "ml-explore/mlx-examples"
DEFAULT_MODEL = "facebook/sam-vit-base"


def _request(url: str):
    return urllib.request.Request(url, headers={"User-Agent": "Tator-MLX-SAM-setup"})


def _resolve_ref(ref: str) -> str:
    encoded = urllib.parse.quote(ref, safe="")
    url = f"https://api.github.com/repos/{UPSTREAM_REPO}/commits/{encoded}"
    with urllib.request.urlopen(_request(url), timeout=30) as response:
        payload = json.load(response)
    sha = str(payload.get("sha") or "")
    if len(sha) != 40:
        raise RuntimeError(f"Unable to resolve mlx-examples ref {ref!r}")
    return sha


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(_request(url), timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _safe_extract_segment_anything(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        readme = next(
            (member for member in members if member.name.endswith("/segment_anything/README.md")),
            None,
        )
        if readme is None:
            raise RuntimeError("mlx-examples archive has no segment_anything source")
        prefix = readme.name[: -len("README.md")]
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        for member in members:
            if not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix) :]
            if not relative:
                continue
            target = (destination / relative).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"Unsupported archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read archive entry: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _source_ready(path: Path) -> bool:
    return (
        (path / "convert.py").is_file()
        and (path / "segment_anything" / "sam.py").is_file()
        and (path / "segment_anything" / "predictor.py").is_file()
    )


def _model_ready(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the official MLX SAM source and convert SAM ViT-B into Tator's ignored cache."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mlx-examples-ref", default="main")
    parser.add_argument("--hf-model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    source_dir = root / "mlx-examples" / "segment_anything"
    model_dir = root / "models" / "sam-vit-base"
    resolved_sha = None

    if args.force and source_dir.exists():
        shutil.rmtree(source_dir)
    if not _source_ready(source_dir):
        resolved_sha = _resolve_ref(args.mlx_examples_ref)
        archive_url = f"https://github.com/{UPSTREAM_REPO}/archive/{resolved_sha}.tar.gz"
        with tempfile.TemporaryDirectory(prefix="tator-mlx-sam-") as temp_dir:
            archive_path = Path(temp_dir) / "mlx-examples.tar.gz"
            print(f"Downloading mlx-examples {resolved_sha[:12]}...")
            _download(archive_url, archive_path)
            if source_dir.exists():
                shutil.rmtree(source_dir)
            _safe_extract_segment_anything(archive_path, source_dir)

    if args.force and model_dir.exists():
        shutil.rmtree(model_dir)
    if not _model_ready(model_dir):
        print(f"Converting {args.hf_model} for MLX...")
        subprocess.run(
            [
                sys.executable,
                str(source_dir / "convert.py"),
                "--hf-path",
                args.hf_model,
                "--mlx-path",
                str(model_dir),
            ],
            check=True,
            cwd=source_dir,
        )

    if not _source_ready(source_dir) or not _model_ready(model_dir):
        raise RuntimeError("MLX SAM setup did not produce a complete runtime")

    manifest = {
        "mlx_examples_repo": f"https://github.com/{UPSTREAM_REPO}",
        "mlx_examples_ref": args.mlx_examples_ref,
        "mlx_examples_sha": resolved_sha,
        "hf_model": args.hf_model,
        "model_path": str(model_dir),
        "source_path": str(source_dir),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "setup.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("MLX SAM is ready.")
    print(f"SAM_MLX_MODEL_PATH={model_dir}")
    print(f"SAM_MLX_ROOT={source_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
