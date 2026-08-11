# Environment Setup

Tator uses separate Python environments for Apple Silicon inference, general
Linux use, and the pinned Falcon CUDA stack.

## Poetry setup

Install the repository setup command:

```bash
poetry install --only-root
```

Choose a profile:

```bash
poetry run tator-setup macos
poetry run tator-setup linux
poetry run tator-setup falcon-cu118
```

Useful options:

```bash
poetry run tator-setup macos --dry-run
poetry run tator-setup linux --dev
poetry run tator-setup falcon-cu118 --venv-dir .venv-falcon
poetry run tator-setup macos --recreate
```

The macOS profile uses `.venv-macos`. Linux profiles use `.venv` unless
`--venv-dir` is provided.

Equivalent shell helpers are available:

```bash
tools/setup_venv_macos_inference.sh
bash tools/setup_venv_falcon_cu118.sh
```

## Falcon CUDA profile

The pinned Falcon profile targets Linux x86_64 with:

- Python 3.10 or 3.11
- NVIDIA driver 520.61.05 or newer
- PyTorch 2.7.1 with CUDA 11.8 wheels
- TorchVision 0.22.1
- TorchAudio 2.7.1
- Transformers 4.57.1
- Accelerate 1.12.0
- NumPy 1.26.0

Install it with:

```bash
poetry run tator-setup falcon-cu118
```

Add `--dev` for test and lint dependencies. Without Poetry:

```bash
bash tools/setup_venv_falcon_cu118.sh
INSTALL_DEV=1 bash tools/setup_venv_falcon_cu118.sh
```

PyTorch wheels include their CUDA runtime. A local CUDA toolkit is needed only
for building custom CUDA extensions. Use
`constraints/falcon-cu118.txt` when installing additions into this profile.

## Verify

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

For macOS runtime checks, see
[macOS Inference Setup](macos_inference_setup.md).
