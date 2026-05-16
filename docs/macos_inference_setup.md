# macOS Inference Runtime

This path is for running the annotation-assistance backend on Apple Silicon. It targets interactive inference only:

- CLIP classifier assistance
- SAM/SAM3 prompt assistance
- YOLO inference
- RF-DETR inference
- Qwen captions, structured VLM inference, and prepass context

Training is intentionally out of scope for this Mac path.

## Current Architecture

The backend is still a single FastAPI app exported by `app/__init__.py` and wired in `localinferenceapi.py`.

Runtime loading happens in these places:

- CLIP: `localinferenceapi.py` loads `clip.load(...)` into the global classifier backbone.
- SAM1 interactive prompts: `localinferenceapi.py` builds `SamPredictor` slots.
- SAM3 text/visual prompts: `services/sam3_runtime.py` resolves the device and caches the text runtime.
- YOLO: `services/detectors.py` calls `model.predict(...)`; `localinferenceapi.py` now supplies a resolved inference device.
- RF-DETR: `services/detectors.py` constructs the RF-DETR model with a resolved device string.
- Qwen: `services/qwen_runtime.py` handles the Transformers/PyTorch path, while `services/qwen_mlx.py` selects MLX-VLM on Apple Silicon when available.

## Acceleration Strategy

The Mac backend uses two acceleration families:

- PyTorch MPS for CLIP, SAM/SAM3, YOLO, and RF-DETR.
- MLX-VLM for Qwen3-VL inference on Apple Silicon, with Transformers/PyTorch as the fallback.

Reason:

- CLIP, SAM, YOLO, and RF-DETR are already loaded as PyTorch models in this codebase, so MPS is the shortest path for the existing annotation flow.
- Qwen3-VL has maintained quantized MLX community builds, so Apple Silicon VLM inference can use native MLX without porting detector/SAM weights.
- Adapter checkpoints still use the Transformers path because they are tied to Hugging Face/PyTorch loading.

New environment controls:

```bash
TATOR_INFERENCE_DEVICE=auto   # auto|mps|cpu|cuda...
TATOR_ALLOW_MPS=1
PYTORCH_ENABLE_MPS_FALLBACK=1 # route unsupported MPS ops to CPU
YOLO_INFER_DEVICE=auto        # optional per-runtime override
RFDETR_INFER_DEVICE=auto      # optional per-runtime override
SAM3_DEVICE=auto
QWEN_DEVICE=auto
QWEN_INFERENCE_PLATFORM=auto # auto|mlx_vlm|transformers
QWEN_MLX_MODEL_NAME=mlx-community/Qwen3-VL-4B-Instruct-4bit
QWEN_MLX_DEFAULT_QUANTIZATION=4bit
```

On Apple Silicon with MPS available, `TATOR_INFERENCE_DEVICE=auto` resolves to
`mps` for PyTorch-backed runtimes. `QWEN_INFERENCE_PLATFORM=auto` resolves to
`mlx_vlm` when `mlx-vlm` imports successfully and no adapter checkpoint is
active; otherwise it falls back to `transformers`.

## Setup

Use Python 3.11. Do not use the existing `.venv` if it was created with Python 3.13+.

```bash
tools/setup_venv_macos_inference.sh
cp .env.macos.example .env.macos
tools/run_macos_backend.sh
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

Qwen runtime smoke:

```bash
curl http://127.0.0.1:8000/qwen/settings
curl http://127.0.0.1:8000/qwen/status
```

On a working Apple Silicon MLX setup, `/qwen/status` should report
`platform: "mlx_vlm"`, `device: "mlx_vlm"`, and an
`effective_model_name` under `mlx-community/`.

## Qwen MLX-VLM

The macOS requirements install `mlx` and `mlx-vlm` by default. The backend
exposes MLX model options through `/qwen/settings` and the browser UI under
**Backend Config -> Qwen Runtime (advanced)**.

Useful environment settings:

```bash
QWEN_INFERENCE_PLATFORM=auto
QWEN_MLX_MODEL_NAME=mlx-community/Qwen3-VL-4B-Instruct-4bit
QWEN_MLX_DEFAULT_QUANTIZATION=4bit
```

Runtime selection rules:

- `auto` uses MLX-VLM on Apple Silicon when `mlx-vlm` is importable.
- `auto` falls back to Transformers when a trained/adapter Qwen checkpoint is active.
- `mlx_vlm` forces MLX-VLM and returns a clear 503 error if the package cannot load.
- `transformers` forces the existing Hugging Face/PyTorch path.

The UI lists the quantized Qwen3-VL options from the `mlx-community/qwen3-vl`
collection, including 2B, 4B, 8B, 30B-A3B, 32B, and available 235B-A22B
variants. Choose a model that fits local RAM/VRAM; the list is capability
surface, not a guarantee that every model is practical on every Mac.

## SAM3 Notes

The macOS inference requirements install upstream SAM3 directly from GitHub. First model load can download the `facebook/sam3` checkpoint and assets from Hugging Face, so expect a large cache fill before the first prompt returns.

Upstream SAM3 currently imports a few CUDA/Triton helper modules even when the runtime target is CPU or MPS. The backend applies macOS compatibility patches during import so the image model can load and then moves it to MPS. Keep `PYTORCH_ENABLE_MPS_FALLBACK=1` set; PyTorch will run unsupported MPS ops on CPU when needed.

## Known Limits

- YOLO is expected to work through Ultralytics `device=mps`.
- CLIP and SAM1 use PyTorch MPS directly.
- SAM3 visual prompts have been smoke-tested on MPS. Some internal ops still fall back to CPU; set `SAM3_DEVICE=cpu` when debugging SAM3-specific runtime issues.
- RF-DETR receives `device=mps`, but package-level MPS coverage must be verified with real weights. If it fails on unsupported ops, set `RFDETR_INFER_DEVICE=cpu` while keeping YOLO/SAM/CLIP on MPS.
- Qwen MLX-VLM does not stream tokens yet; streaming endpoints return the final generated text once the MLX call completes.
- Qwen adapter checkpoints use Transformers/PyTorch, not MLX-VLM.
