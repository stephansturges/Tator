# Environment Setup

Tator uses profile-specific Python environments. A single locked Poetry
environment is not a good fit for every path because the macOS MLX runtime,
general Linux training stack, and Falcon CUDA stack have intentionally different
constraints. Poetry is therefore used as the command front door, while the
profile installs still land in the virtualenvs that the backend launch scripts
expect.

## Recommended Setup Front Door

Install Poetry once, then install only the repo setup command:

```bash
poetry install --only-root
```

Then choose the profile that matches the machine:

```bash
# Apple Silicon inference + Qwen MLX adapter training
poetry run tator-setup macos

# General Linux backend/training stack
poetry run tator-setup linux

# Pinned Falcon CUDA 11.8 stack
poetry run tator-setup falcon-cu118
```

Useful options:

```bash
poetry run tator-setup macos --dry-run
poetry run tator-setup linux --dev
poetry run tator-setup falcon-cu118 --venv-dir .venv-falcon
poetry run tator-setup macos --recreate
```

The generated environments are:

- `.venv-macos` for the Apple Silicon profile.
- `.venv` for the default Linux and Falcon profiles, unless `--venv-dir` is provided.

The older shell commands remain available and now delegate to the same setup
implementation:

```bash
tools/setup_venv_macos_inference.sh
bash tools/setup_venv_falcon_cu118.sh
```

This repo supports a broad dependency range, but the Falcon automatic-labeling
path is more sensitive than the rest of the stack. This document defines the
recommended GPU setup for Falcon work and explains when a CUDA or driver update
is actually needed.

## Short Answer

For Falcon on this repo, update the Python environment first. Do not start by
installing a local CUDA toolkit.

Why:
- PyTorch pip wheels already ship the CUDA runtime they need.
- `nvcc` is not required unless you are building custom CUDA extensions.
- Falcon's current failure mode in this repo is tied to PyTorch FlexAttention
  behavior, not a missing local CUDA toolkit.

## Recommended Falcon GPU Stack

This is the first stack to try for Falcon automatic-labeling on Linux x86_64:

- Python `3.10` or `3.11`
- NVIDIA driver `>= 520.61.05`
- PyTorch `2.7.1`
- TorchVision `0.22.1`
- TorchAudio `2.7.1`
- PyTorch wheel flavor: `cu118`
- Transformers `4.57.1`
- Accelerate `1.12.0`
- Hugging Face Hub `0.36.0`
- Tokenizers `0.22.1`
- NumPy `1.26.0`
- `fsspec` `2025.10.0`

The repo includes a reproducible setup script for this stack:

```bash
poetry run tator-setup falcon-cu118
```

To include dev tools too:

```bash
poetry run tator-setup falcon-cu118 --dev
```

Without Poetry, `bash tools/setup_venv_falcon_cu118.sh` and
`INSTALL_DEV=1 bash tools/setup_venv_falcon_cu118.sh` perform the same setup.

## Why `cu118`

PyTorch 2.7.1 is the important upgrade for Falcon. The Falcon model card only
states `torch>=2.5`, but in practice our repo hit a FlexAttention limitation on
`2.5.1+cu121`. PyTorch 2.7 explicitly adds newer FlexAttention inference
support, including GQA-related improvements.

Using the `cu118` wheels is the least disruptive way to get that newer PyTorch:

- it avoids a system CUDA toolkit install
- it avoids a driver upgrade on machines that already satisfy the CUDA 11.8
  driver floor
- it still gives us the newer PyTorch runtime and FlexAttention code

## When a Driver or CUDA Update Is Actually Needed

You only need to move the system driver if you choose a newer wheel set whose
driver floor exceeds your current driver.

Useful reference points:

- PyTorch `2.7.1` offers official wheels for:
  - `cu118`
  - `cu126`
  - `cu128`
- NVIDIA's CUDA 11.8 driver floor is `>= 520.61.05`
- NVIDIA's CUDA 12.6 toolkit driver is `>= 560.28.03`

Practical recommendation:

- If your driver is already `>= 520.61.05`, try `2.7.1+cu118` first.
- Only consider moving to `cu126` or newer if:
  - you have already confirmed the Falcon runtime still needs newer PyTorch
    behavior beyond `2.7.1`, or
  - you are upgrading the driver for other reasons anyway.

Do not install a local CUDA toolkit just to use PyTorch wheels.

## Repo Dependency Notes

The repo's `requirements.txt` stays intentionally broad because most features do
not need a single locked stack. For Falcon work, use:

- `tools/setup_venv_falcon_cu118.sh`
- `constraints/falcon-cu118.txt`

This keeps the general repo install flexible while giving Falcon a known-good
path.

## Known Compatibility Notes

- `sam3` currently declares `numpy==1.26`, so we pin NumPy to `1.26.0` here.
- `datasets 4.4.2` expects `fsspec<=2025.10.0`, so we pin that too.
- Falcon in this repo still uses local compatibility patches in
  `services/falcon_perception.py` because its Hugging Face remote code does not
  line up cleanly with every newer PyTorch attention path.

## Validation

After setup:

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
print("devices", torch.cuda.device_count())
PY
```

Expected outcome for the recommended path:

- `torch 2.7.1+cu118`
- CUDA available on the GPU machine

## Optional Newer-Wheel Path

If you intentionally want newer CUDA wheels later:

- `cu126` is the next candidate
- treat that as a driver-upgrade decision, not a Falcon-first requirement
- document the new driver floor before switching the repo's recommended setup
