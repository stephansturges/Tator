# macOS Inference Runtime

This path is for running the annotation-assistance backend on Apple Silicon. It targets interactive inference only:

- CLIP classifier assistance
- SAM/SAM3 prompt assistance
- YOLO inference
- RF-DETR inference
- optional Qwen captioning later

Training is intentionally out of scope for this Mac path.

## Current Architecture

The backend is still a single FastAPI app exported by `app/__init__.py` and wired in `localinferenceapi.py`.

Runtime loading happens in these places:

- CLIP: `localinferenceapi.py` loads `clip.load(...)` into the global classifier backbone.
- SAM1 interactive prompts: `localinferenceapi.py` builds `SamPredictor` slots.
- SAM3 text/visual prompts: `services/sam3_runtime.py` resolves the device and caches the text runtime.
- YOLO: `services/detectors.py` calls `model.predict(...)`; `localinferenceapi.py` now supplies a resolved inference device.
- RF-DETR: `services/detectors.py` constructs the RF-DETR model with a resolved device string.
- Qwen: `services/qwen_runtime.py` already resolves CUDA, MPS, then CPU for captioning.

## Acceleration Strategy

The first Mac backend uses PyTorch MPS, not a full MLX model port.

Reason:

- CLIP, SAM, YOLO, and RF-DETR are already loaded as PyTorch models in this codebase.
- PyTorch MPS is the shortest path to accelerating the existing annotation flow without changing model formats.
- MLX is left as a follow-up VLM experiment, but porting detector/SAM weights to MLX would require separate model-specific adapters.

New environment controls:

```bash
TATOR_INFERENCE_DEVICE=auto   # auto|mps|cpu|cuda...
TATOR_ALLOW_MPS=1
PYTORCH_ENABLE_MPS_FALLBACK=1 # route unsupported MPS ops to CPU
YOLO_INFER_DEVICE=auto        # optional per-runtime override
RFDETR_INFER_DEVICE=auto      # optional per-runtime override
SAM3_DEVICE=auto
QWEN_DEVICE=auto
```

On Apple Silicon with MPS available, `auto` resolves to `mps`.

## Setup

Use Python 3.11. Do not use the existing `.venv` if it was created with Python 3.13+.

```bash
tools/setup_venv_macos_inference.sh
cp .env.macos.example .env.macos
tools/run_macos_backend.sh
```

Optional VLM experiment packages:

```bash
INSTALL_VLM=1 tools/setup_venv_macos_inference.sh
```

Then open:

```text
ybat-master/ybat.html
```

The UI defaults to `http://localhost:8000`.

## Verification

Basic backend import and device smoke:

```bash
source .venv-macos/bin/activate
TATOR_SKIP_CLIP_LOAD=1 python - <<'PY'
import torch
import localinferenceapi
print("routes", len(localinferenceapi.app.routes))
print("mps", torch.backends.mps.is_available())
print(localinferenceapi._gpu_status_payload())
PY
```

Health endpoint after the server starts:

```bash
curl http://127.0.0.1:8000/system/health_summary
```

The GPU status payload now reports `mps.available`, `preferred_inference_device`, and `inference_backend`.

## SAM3 Notes

The macOS inference requirements install upstream SAM3 directly from GitHub. First model load can download the `facebook/sam3` checkpoint and assets from Hugging Face, so expect a large cache fill before the first prompt returns.

Upstream SAM3 currently imports a few CUDA/Triton helper modules even when the runtime target is CPU or MPS. The backend applies macOS compatibility patches during import so the image model can load and then moves it to MPS. Keep `PYTORCH_ENABLE_MPS_FALLBACK=1` set; PyTorch will run unsupported MPS ops on CPU when needed.

## Known Limits

- YOLO is expected to work through Ultralytics `device=mps`.
- CLIP and SAM1 use PyTorch MPS directly.
- SAM3 visual prompts have been smoke-tested on MPS. Some internal ops still fall back to CPU; set `SAM3_DEVICE=cpu` when debugging SAM3-specific runtime issues.
- RF-DETR receives `device=mps`, but package-level MPS coverage must be verified with real weights. If it fails on unsupported ops, set `RFDETR_INFER_DEVICE=cpu` while keeping YOLO/SAM/CLIP on MPS.
- Qwen captioning can already use PyTorch MPS. MLX/`mlx-vlm` can be installed with `INSTALL_VLM=1` for later adapter work, but it is not wired into the backend yet.
