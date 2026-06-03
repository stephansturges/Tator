# macOS Inference Runtime

This path is for running the annotation-assistance backend on Apple Silicon. It targets interactive inference plus Qwen MLX adapter jobs:

- CLIP classifier assistance
- SAM/SAM3 prompt assistance
- YOLO inference
- RF-DETR inference
- Qwen captions, structured VLM inference, prepass context, and MLX LoRA adapter training

Full detector/SAM training remains Linux/CUDA-first. The Mac path supports Qwen MLX adapter training for small enough local models.

## Current Architecture

The backend is still a single FastAPI app exported by `app/__init__.py` and wired in `localinferenceapi.py`.

Runtime loading happens in these places:

- CLIP: `localinferenceapi.py` loads `clip.load(...)` into the global classifier backbone.
- DINOv3: `services/mlx_dinov3.py` can route ViT DINOv3 encoding through a
  Swift/MLX worker when converted weights are present, otherwise Torch/MPS is
  used.
- SAM1 interactive prompts: `localinferenceapi.py` builds `SamPredictor` slots.
  `SAM1_BACKEND=auto` can use the MLX SAM adapter on Apple Silicon when a
  converted MLX SAM model and the `mlx-examples` SAM package are configured.
- SAM3 text/visual prompts: `services/sam3_runtime.py` resolves the device and caches the text runtime.
- YOLO: `services/detectors.py` calls `model.predict(...)`; `localinferenceapi.py` now supplies a resolved inference device.
- RF-DETR: `services/detectors.py` constructs the RF-DETR model with a resolved device string.
- Qwen: `services/qwen_runtime.py` handles the Transformers/PyTorch path, while `services/qwen_mlx.py` selects MLX-VLM on Apple Silicon when available.

## Acceleration Strategy

The Mac backend uses two acceleration families:

- PyTorch MPS for CLIP, SAM3, YOLO, and RF-DETR.
- Optional MLX SAM1 for interactive click/box annotation when converted SAM
  weights are available locally.
- MLX-DINOv3 for DINOv3 ViT embedding jobs when the worker and converted model
  cache are available.
- MLX-VLM for Qwen3-VL inference on Apple Silicon, with Transformers/PyTorch as the fallback.

Reason:

- CLIP, SAM3, YOLO, and RF-DETR are already loaded as PyTorch models in this codebase, so MPS is the shortest path for those runtimes.
- SAM1 can use MLX through the `mlx-examples` Segment Anything implementation;
  the adapter falls back to the existing PyTorch SAM1 path unless `SAM1_BACKEND=mlx`
  is explicitly requested.
- Qwen3-VL has maintained quantized MLX community builds, so Apple Silicon VLM inference and Qwen adapter training can use native MLX without porting detector/SAM weights.
- Adapter checkpoints preserve their runtime family: Transformers adapters load through PEFT, while MLX adapters load through MLX-VLM.

New environment controls:

```bash
TATOR_INFERENCE_DEVICE=auto   # auto|mps|cpu|cuda...
TATOR_ALLOW_MPS=1
PYTORCH_ENABLE_MPS_FALLBACK=1 # route unsupported MPS ops to CPU
YOLO_INFER_DEVICE=auto        # optional per-runtime override
RFDETR_INFER_DEVICE=auto      # optional per-runtime override
SAM3_DEVICE=auto
SAM1_BACKEND=auto             # auto|torch|mlx
SAM_MLX_MODEL_PATH=           # converted MLX SAM model dir with config.json/model.safetensors
SAM_MLX_ROOT=                 # mlx-examples/segment_anything checkout
DINOV3_BACKEND=auto           # auto|torch|mlx
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

Recommended Poetry front door:

```bash
poetry install --only-root
poetry run tator-setup macos
cp .env.macos.example .env.macos
tools/run_macos_backend.sh
```

After setup, the daily backend start command is:

```bash
tools/run_macos_backend.sh
```

From another directory, `cd` to your clone first:

```bash
cd /path/to/Tator && tools/run_macos_backend.sh
```

Equivalent direct wrapper, for machines without Poetry:

```bash
tools/setup_venv_macos_inference.sh
cp .env.macos.example .env.macos
tools/run_macos_backend.sh
```

Then open:

```text
http://127.0.0.1:8000/
```

The backend listens on `http://127.0.0.1:8000` and serves the browser UI at `/`
and `/tator.html`. The old `/ybat.html` URL redirects to `/tator.html`.

For frontend development, you can run a separate static UI server from the repo
root:

```bash
python3 -m http.server 8080 -d ybat-master
```

Then open:

```text
http://127.0.0.1:8080/tator.html
```

The UI still talks to the backend at `http://localhost:8000` by default.

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

MLX-DINOv3 worker smoke:

```bash
tools/build_mlx_dinov3_worker.sh
tools/convert_mlx_dinov3_model.sh facebook/dinov3-vitb16-pretrain-lvd1689m
source .venv-macos/bin/activate
python - <<'PY'
from services.mlx_dinov3 import mlx_dinov3_status
print(mlx_dinov3_status("facebook/dinov3-vitb16-pretrain-lvd1689m").to_dict())
PY
```

On this machine, `auto` resolves to MLX after those two commands because the
worker exists and the converted model is cached under
`uploads/model_cache/mlx_dinov3/`. If either asset is missing, `auto` falls back
to Torch/MPS before a job starts.

MLX SAM1 annotation smoke, after cloning `mlx-examples` and converting a SAM
checkpoint with its `segment_anything/convert.py` script:

```bash
source .venv-macos/bin/activate
SAM1_BACKEND=mlx \
SAM_MLX_MODEL_PATH=/path/to/sam-vit-base-mlx \
SAM_MLX_ROOT=/path/to/mlx-examples/segment_anything \
python - <<'PY'
import localinferenceapi as api
print(api._system_health_summary()["models"]["sam1"])
PY
```

The backend only switches the annotation SAM1 slots to MLX after restart with
those environment variables. `SAM1_BACKEND=auto` falls back to Torch/MPS when
the MLX assets are absent. It also probes the MLX runtime before advertising the
adapter as available. In headless, sandboxed, or otherwise non-Metal sessions,
installed `mlx` packages can still raise `No Metal device available`; those
sessions now report `mlx_runtime_unavailable` and remain on the Torch SAM1 path
instead of selecting an unusable MLX predictor.

Qwen runtime smoke:

```bash
curl http://127.0.0.1:8000/qwen/settings
curl http://127.0.0.1:8000/qwen/status
```

On a working Apple Silicon MLX setup, `/qwen/status` should report
`platform: "mlx_vlm"`, `device: "mlx_vlm"`, and an
`effective_model_name` under `mlx-community/`.

## Qwen MLX-VLM

The setup script installs `mlx`, the direct MLX-VLM runtime dependencies, and
then `mlx-vlm==0.3.9` from `requirements-macos-vlm.txt` with `--no-deps`. This
keeps the main macOS environment on Transformers 4.57 and the SAHI-compatible
OpenCV line while using the newest Qwen3-capable MLX-VLM 0.3.x release. The backend
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
variants. It also includes compatible abliterated MLX builds from EZCon,
alexgusevski, nightmedia, introvoyz041, veeceey, and Goekdeniz-Guelmez where
those repos expose MLX-format safetensors. Choose a model that fits local
RAM/VRAM; the list is capability surface, not a guarantee that every model is
practical on every Mac.

CUDA machines should use the Transformers model registry in **Qwen Models**.
That registry exposes the official full and FP8 Qwen3-VL checkpoints, curated
compressed-tensors/AWQ-style and GPTQ 4-bit checkpoints, and compatible
abliterated Transformer checkpoints from Huihui, Prithiv, and quantized
community conversions. Quantized activation uses the selected checkpoint for
inference when the CUDA dependency stack supports it. Dense and MoE
Transformers entries can be used for LoRA/QLoRA training; quantized CUDA
selections resolve to the matching full checkpoint before training and apply
bitsandbytes QLoRA there. MLX entries train through MLX-VLM on Apple Silicon,
including quantized checkpoints for QLoRA-style adapter training. FP8 requires
compatible NVIDIA hardware.

## MLX-DINOv3

The optional DINOv3 worker is a SwiftPM package in
`tools/mlx_dinov3_worker/`, pinned to `vincentamato/MLXDINOv3` commit
`3122d7905cca21012b4c249e8ddad19ff78f54bc`. It supports ViT DINOv3 checkpoints;
ConvNeXt-style DINOv3 models remain on Torch.

Build and convert:

```bash
tools/build_mlx_dinov3_worker.sh
tools/convert_mlx_dinov3_model.sh facebook/dinov3-vitb16-pretrain-lvd1689m
```

`swift build` can produce the binaries, but MLX also needs `mlx.metallib` next
to them. The build script compiles the MLX generated Metal sources into that
metallib. If the normal Xcode `metal` wrapper reports a missing Metal Toolchain,
run `xcodebuild -downloadComponent MetalToolchain`; on some systems the
downloaded cryptex must be mounted before the script can use
`/Volumes/MetalToolchainCryptex/Metal.xctoolchain/usr/bin/metal`.

Selection rules:

- `DINOV3_BACKEND=auto` uses MLX only when the platform is Apple Silicon and
  both the worker and converted model are present.
- `DINOV3_BACKEND=torch` forces the existing Hugging Face/Torch path.
- `DINOV3_BACKEND=mlx` requires MLX and fails clearly if the worker/model is
  unavailable.
- A job never silently switches between MLX and Torch after it starts.

Runtime coverage:

- Data Ingestion reference profile training, candidate scoring, pooled DINOv3,
  local SALAD-over-DINOv3, and local Vendi patch scoring use the resolver.
- Class Split Explorer and active DINOv3 classifier inference use the resolver.
- Train Class Predictor uses the resolver for frozen DINOv3 feature extraction;
  metadata records `dinov3_backend` for trained artifacts.

Data Ingestion CPU-side controls:

- `DATA_INGESTION_MEDIA_PREPARE_WORKERS` defaults to `8`; the Data Ingestion UI
  can override it per job for video frame extraction, image preparation, and
  reference-view augmentation.
- `DATA_INGESTION_WORKER_MAX` caps the UI/env worker value, defaulting to `16`
  or the local CPU count, whichever is lower.
- `LOCAL_SALAD_TRAIN_VIEW_MAX_SIDE` defaults to `384`; profile-training views
  are augmented at this bound before the DINOv3 worker resizes them to its
  native `224` input.
- Matching saved reference profiles are reused by training signature. Set
  `DATA_INGESTION_FORCE_PROFILE_REBUILD=1` or send `force_rebuild_profile` in
  the job manifest when a deliberate retrain is needed.

Validation on the local ViT-B checkpoint:

- Worker smoke returned CLS `[1, 768]` and patch tokens `[1, 196, 768]`.
- Two-image parity against Torch/MPS showed CLS cosine minimum `0.999994` and
  patch-token cosine minimum `0.999998`.
- A 32-image synthetic throughput check measured MLX at about `342-361 imgs/s`
  for batches 4-16 versus Torch/MPS at about `180-212 imgs/s`.

## SAM3 Notes

The macOS inference requirements install upstream SAM3 directly from GitHub. First model load can download the `facebook/sam3` checkpoint and assets from Hugging Face, so expect a large cache fill before the first prompt returns.

Upstream SAM3 currently imports a few CUDA/Triton helper modules even when the runtime target is CPU or MPS. The backend applies macOS compatibility patches during import so the image model can load and then moves it to MPS. Keep `PYTORCH_ENABLE_MPS_FALLBACK=1` set; PyTorch will run unsupported MPS ops on CPU when needed.

## Known Limits

- YOLO is expected to work through Ultralytics `device=mps`.
- CLIP uses PyTorch MPS. SAM1 uses PyTorch MPS unless `SAM1_BACKEND` selects
  the optional MLX adapter.
- SAM3 visual prompts have been smoke-tested on MPS. Some internal ops still fall back to CPU; set `SAM3_DEVICE=cpu` when debugging SAM3-specific runtime issues.
- RF-DETR receives `device=mps`, but package-level MPS coverage must be verified with real weights. If it fails on unsupported ops, set `RFDETR_INFER_DEVICE=cpu` while keeping YOLO/SAM/CLIP on MPS.
- Qwen MLX-VLM does not stream tokens yet; streaming endpoints return the final generated text once the MLX call completes.
- Qwen adapter checkpoints preserve their training runtime. Transformers adapters load through PEFT; MLX adapters load through MLX-VLM with their base model.
- Qwen3-VL MoE adapter training is wired through Transformers `Qwen3VLMoeForConditionalGeneration`, but practical runs need very large CUDA memory or QLoRA/distributed setups.
- `pip check` will report the intentional `mlx-vlm==0.3.9` OpenCV metadata mismatch in `.venv-macos`; SAHI requires OpenCV <=4.11, and MLX-VLM works here with the SAHI-compatible OpenCV installed by the setup script.
