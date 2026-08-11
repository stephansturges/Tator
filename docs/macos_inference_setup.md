# macOS Inference Setup

Tator supports interactive inference on Apple Silicon. Detector and segmentation
training remain CUDA-first; local Qwen adapter training is available for models
that fit the machine.

## Install

Use Python 3.11.

```bash
poetry install --only-root
poetry run tator-setup macos
cp .env.macos.example .env.macos
```

Without Poetry:

```bash
tools/setup_venv_macos_inference.sh
cp .env.macos.example .env.macos
```

Start the application from the repository root:

```bash
tools/run_macos_backend.sh
```

Open `http://127.0.0.1:8000/`. Set `PORT` before launching to choose a
different port.

## Runtime selection

PyTorch-backed CLIP, SAM3, YOLO, and RF-DETR inference can use MPS. Qwen uses
MLX-VLM when available on Apple Silicon and falls back to Transformers when the
selected configuration requires it. SAM1 and DINOv3 have optional MLX paths.

Common settings:

```bash
TATOR_INFERENCE_DEVICE=auto
TATOR_ALLOW_MPS=1
PYTORCH_ENABLE_MPS_FALLBACK=1

YOLO_INFER_DEVICE=auto
RFDETR_INFER_DEVICE=auto
SAM3_DEVICE=auto

SAM1_BACKEND=auto
SAM_MLX_MODEL_PATH=
SAM_MLX_ROOT=

DINOV3_BACKEND=auto

QWEN_DEVICE=auto
QWEN_INFERENCE_PLATFORM=auto
QWEN_MLX_MODEL_NAME=
QWEN_MLX_CAPTION_MODEL_NAME=
QWEN_MODEL_NAME=
QWEN_TRAINING_DEFAULT_MODEL=
QWEN_MLX_DEFAULT_QUANTIZATION=4bit
```

`auto` selects the available accelerated runtime and falls back before a job
starts. An active job does not switch embedding backends midway through a run.
Model selectors expose only entries whose catalog metadata matches the requested
capability; inference-only entries do not appear in training selectors.

## Verify the backend

After startup:

```bash
curl http://127.0.0.1:8000/system/health_summary
curl http://127.0.0.1:8000/qwen/settings
curl http://127.0.0.1:8000/qwen/status
```

The health response reports the selected inference device and the availability
of optional runtimes.

## Optional MLX-DINOv3

Build the Swift worker and convert a supported ViT checkpoint:

```bash
tools/build_mlx_dinov3_worker.sh
tools/convert_mlx_dinov3_model.sh facebook/dinov3-vitb16-pretrain-lvd1689m
```

Converted models are stored under `uploads/model_cache/mlx_dinov3/`.
`DINOV3_BACKEND=auto` uses MLX only when both the worker and converted model
are available. Use `torch` or `mlx` to force a backend.

## Optional MLX SAM1

Configure a converted SAM model and the MLX Segment Anything package:

```bash
SAM1_BACKEND=mlx
SAM_MLX_MODEL_PATH=/path/to/sam-mlx
SAM_MLX_ROOT=/path/to/mlx-examples/segment_anything
```

Restart after changing these values. `SAM1_BACKEND=auto` falls back to the
PyTorch path when the MLX assets or Metal runtime are unavailable.

## Qwen

The browser runtime controls are under **Backend Config -> Qwen Runtime
(advanced)**. General inference, captioning, and training defaults are separate
so a large reviewer model is not implicitly used for every caption pass or
training job.

Qwen caption controls are grouped by:

- image scope and context;
- caption and editor model selection;
- generation options and output guards.

**Same as captioning** keeps the caption model for editor passes. **Auto editor
model** selects a compact editor. An explicit model ID forces that editor model.

MLX adapters load through MLX-VLM; Transformers adapters load through PEFT.
Quantized and full checkpoints remain subject to the capabilities declared in
the model catalog.

## Troubleshooting

- If the requested port is occupied, the launcher reports the listener and
  exits. Choose another `PORT` or stop the intended listener explicitly.
- If an MPS operation is unsupported, keep
  `PYTORCH_ENABLE_MPS_FALLBACK=1` or force that runtime to CPU.
- If MLX reports no Metal device, use the corresponding `auto` or PyTorch
  backend and confirm the process is running in a Metal-capable session.
- Initial model use can populate the Hugging Face cache substantially. Select
  only models that fit available memory and disk.
- MLX Qwen endpoints currently return the completed text after generation
  rather than token-by-token output.
