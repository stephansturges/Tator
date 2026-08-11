"""Profile-aware environment setup for Tator.

Poetry is used as a stable command runner, while this script creates the
profile-specific virtual environments used by the backend launch scripts.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    default_venv: str
    default_python: str


PROFILES = {
    "macos": Profile(
        name="macos",
        description="Apple Silicon inference plus Qwen MLX adapter training",
        default_venv=".venv-macos",
        default_python="python3.11",
    ),
    "linux": Profile(
        name="linux",
        description="General Linux backend and training environment",
        default_venv=".venv",
        default_python="python3.11",
    ),
    "falcon-cu118": Profile(
        name="falcon-cu118",
        description="Pinned Linux CUDA 11.8 Falcon automatic-labeling stack",
        default_venv=".venv",
        default_python="python3",
    ),
}


def _display_command(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(f"+ {_display_command(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)


def _python_bin(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _create_venv(python_cmd: str, venv_dir: Path, *, recreate: bool, dry_run: bool) -> None:
    if recreate and venv_dir.exists():
        _run(["rm", "-rf", str(venv_dir)], dry_run=dry_run)
    if venv_dir.exists():
        return
    _run([python_cmd, "-m", "venv", str(venv_dir)], dry_run=dry_run)


def _pip_install(
    python: Path,
    args: list[str],
    *,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    _run([str(python), "-m", "pip", "install", *args], dry_run=dry_run, env=env)


def _pip_check(profile: str, python: Path, *, dry_run: bool) -> None:
    command = [str(python), "-m", "pip", "check"]
    print(f"+ {_display_command(command)}")
    if dry_run:
        return
    proc = subprocess.run(command, cwd=ROOT_DIR, text=True, capture_output=True, check=False)
    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    if proc.returncode == 0:
        if output:
            print(output)
        return
    allowed = []
    if profile == "macos":
        allowed.extend(
            [
                "mlx-vlm 0.6.1 has requirement click>=8.2.1",
                "mlx-vlm 0.6.1 has requirement huggingface-hub>=1.0.0",
                "mlx-lm 0.31.3 has requirement transformers>=5.0.0",
                "mlx-vlm 0.6.1 has requirement mlx-audio>=0.4.3",
                "mlx-vlm 0.6.1 has requirement opencv-python>=4.12.0.88",
                "mlx-vlm 0.6.1 has requirement starlette>=1.0.1",
                "mlx-vlm 0.6.1 has requirement transformers>=5.5.0",
                "mlx-vlm 0.6.1 has requirement transformers>=5.0.0",
            ]
        )
    if profile == "falcon-cu118":
        allowed.append("decord 0.6.0 is not supported on this platform")
    if output and all(any(marker in line for marker in allowed) for line in output.splitlines()):
        print(f"Ignoring known {profile} pip check warning:")
        print(output)
        return
    if output:
        print(output, file=sys.stderr)
    raise SystemExit(proc.returncode)


def _install_bootstrap(python: Path, *, dry_run: bool) -> None:
    _pip_install(
        python,
        ["--upgrade", "pip", "wheel", "setuptools<81"],
        dry_run=dry_run,
    )


def _install_macos(args: argparse.Namespace, python: Path) -> None:
    _install_bootstrap(python, dry_run=args.dry_run)
    _pip_install(python, ["-r", "requirements-macos-inference.txt"], dry_run=args.dry_run)
    _pip_install(python, ["--no-deps", "-r", "requirements-macos-vlm.txt"], dry_run=args.dry_run)
    if args.install_local_clip:
        clip_setup = ROOT_DIR / "CLIP" / "setup.py"
        if clip_setup.exists():
            _pip_install(
                python,
                ["--no-build-isolation", "-e", str(ROOT_DIR / "CLIP")],
                dry_run=args.dry_run,
            )
    if not args.skip_pip_check:
        _pip_check("macos", python, dry_run=args.dry_run)


def _install_linux(args: argparse.Namespace, python: Path) -> None:
    _install_bootstrap(python, dry_run=args.dry_run)
    _pip_install(python, ["-r", "requirements.txt"], dry_run=args.dry_run)
    if args.dev:
        _pip_install(python, ["-r", "requirements-dev.txt"], dry_run=args.dry_run)
    if not args.skip_pip_check:
        _pip_check("linux", python, dry_run=args.dry_run)


def _install_falcon(args: argparse.Namespace, python: Path) -> None:
    _install_bootstrap(python, dry_run=args.dry_run)
    _pip_install(
        python,
        [
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "torchaudio==2.7.1",
        ],
        dry_run=args.dry_run,
    )
    _pip_install(
        python,
        ["-r", "requirements.txt", "-c", "constraints/falcon-cu118.txt"],
        dry_run=args.dry_run,
    )
    if args.dev:
        _pip_install(
            python,
            ["-r", "requirements-dev.txt", "-c", "constraints/falcon-cu118.txt"],
            dry_run=args.dry_run,
        )
    if not args.skip_pip_check:
        _pip_check("falcon-cu118", python, dry_run=args.dry_run)


def _finish_message(profile: str, venv_dir: Path) -> None:
    activate = venv_dir / "bin" / "activate"
    print()
    print(f"{profile} environment ready in: {venv_dir}")
    print()
    print("Next:")
    print(f"  source {shlex.quote(str(activate))}")
    if profile == "macos":
        print("  cp .env.macos.example .env.macos")
        print("  tools/run_macos_backend.sh")
    else:
        print("  cp .env.example .env")
        print("  python -m uvicorn app:app --host 0.0.0.0 --port 8000")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up a Tator runtime environment.")
    parser.add_argument(
        "profile",
        choices=sorted(PROFILES),
        help="Environment profile to install.",
    )
    parser.add_argument(
        "--python",
        dest="python_cmd",
        help="Python executable used to create the target virtual environment.",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        help="Target virtual environment directory. Defaults to the profile standard.",
    )
    parser.add_argument("--dev", action="store_true", help="Install requirements-dev.txt.")
    parser.add_argument(
        "--install-local-clip",
        action="store_true",
        help="Install the repo-local CLIP checkout in editable mode when present.",
    )
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the venv.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-pip-check", action="store_true", help="Skip final pip check.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = PROFILES[args.profile]
    venv_dir = (args.venv_dir or Path(profile.default_venv)).resolve()
    python_cmd = args.python_cmd or profile.default_python

    if args.profile == "macos" and args.dev:
        raise SystemExit("The macos profile does not install dev tools; use the linux profile for dev checks.")

    _create_venv(python_cmd, venv_dir, recreate=args.recreate, dry_run=args.dry_run)
    target_python = _python_bin(venv_dir)
    if args.dry_run and not target_python.exists():
        target_python = venv_dir / "bin" / "python"
    if not args.dry_run and not target_python.exists():
        raise SystemExit(f"Virtualenv Python was not created: {target_python}")

    if args.profile == "macos":
        _install_macos(args, target_python)
    elif args.profile == "falcon-cu118":
        _install_falcon(args, target_python)
    else:
        _install_linux(args, target_python)
    _finish_message(args.profile, venv_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
