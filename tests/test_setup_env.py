from tools import setup_env


def test_macos_setup_dry_run_uses_split_mlx_install(tmp_path, capsys):
    venv_dir = tmp_path / ".venv-macos"

    assert (
        setup_env.main(
            [
                "macos",
                "--venv-dir",
                str(venv_dir),
                "--dry-run",
                "--skip-pip-check",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert f"python3.11 -m venv {venv_dir}" in output
    assert "-m pip install -r requirements-macos-inference.txt" in output
    assert "-m pip install --no-deps -r requirements-macos-vlm.txt" in output


def test_falcon_setup_dry_run_uses_cuda_constraints(tmp_path, capsys):
    venv_dir = tmp_path / ".venv-falcon"

    assert (
        setup_env.main(
            [
                "falcon-cu118",
                "--venv-dir",
                str(venv_dir),
                "--dry-run",
                "--skip-pip-check",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "https://download.pytorch.org/whl/cu118" in output
    assert "torch==2.7.1" in output
    assert "-m pip install -r requirements.txt -c constraints/falcon-cu118.txt" in output
