# 🥔 Tator – Local CLIP + SAM Annotation Toolkit

Tator is a single-machine annotation workflow that pairs a clean, fast, simple, web-based frontend with a FastAPI backend to deliver _fast_ bounding-box annotation for images as well as some cool optional automation like class suggestions powered by CLIP and bbox cleanup / auto-suggestion using Segment Anything (SAM). The UI now bundles labeling, CLIP training, and model management in one place so you can iterate on datasets without leaving the browser.

## Lightning-Fast Labeling Modes

### Auto Class Corrector
Drop any tentative label and let CLIP clean it up instantly. Tator double-checks every box and snaps it to the class with the highest confidence so you can move through image stacks at warp speed.

<!-- Add GIF: Auto Class Corrector -->

### Auto Box Refinement
Rough sketches are enough—SAM reshapes your loose bounding boxes into pixel-perfect rectangles while CLIP verifies the class. It feels like spell-check for geometry.

<!-- Add GIF: Auto Box Refinement -->

### One-Click Point-to-Box
Tap once anywhere on the object and SAM conjures a tight box while CLIP names it. Perfect for those “I just need this labeled now” marathons.

<!-- Add GIF: One-Click Point-to-Box -->

### Multi-Point Magic
When objects are tricky, sprinkle a few positive/negative points and let SAM sculpt the exact mask. Queue up new selections immediately—no waiting for the previous mask to land.

<!-- Add GIF: Multi-Point Magic -->

### SAM Preload Boost
Enable preloading to keep the next image warmed up inside SAM. You’ll see progress ticks in the side rail and enjoy near-zero latency when you start pointing.

<!-- Add GIF: SAM Preload Boost -->

## Key Features
- **One-click assists** – auto class, SAM box/point refinements, and multi-point prompts with live progress indicators.
- **SAM 1 & SAM 2** – switch backends at runtime, optionally preload images into SAM to minimise round-trips.
- **Embedded CLIP trainer** – start training jobs from the UI, watch convergence metrics, and reuse cached embeddings across runs.
- **Model switcher** – activate new CLIP + regression pairs without restarting the server; metadata keeps backbone/labelmap in sync.
- **Prometheus metrics** – enable `/metrics` via `.env` for operational visibility.

## Repository Layout
- `app/`, `localinferenceapi.py` – FastAPI app, SAM/CLIP orchestration, training endpoints.
- `ybat-master/` – browser UI (`ybat.html`, CSS/JS, assets).
- `tools/` – reusable training helpers and CLI scripts.
- `uploads/`, `crops/`, `corrected_labels/` – runtime artifacts, embedding cache, and exported crops (ignored by git).
- `AGENTS.md` – contributor handbook and project conventions.

## Prerequisites
- Python 3.10 or newer (3.11+ recommended).
- Optional GPU with CUDA for faster CLIP/SAM inference.
- Model weights: `sam_vit_h_4b8939.pth` (SAM1) plus any desired SAM2 checkpoints/configs.

## Quick Start
1. **Create an environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .\\.venv\\Scripts\\activate
   ```
2. **Install runtime deps**
   ```bash
   pip install -r requirements.txt
   ```
   Torch wheels are hardware-specific; replace `torch`/`torchvision` with the build matching your CUDA/cuDNN stack if needed.
3. **Install dev tooling (optional)**
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```
4. **Fetch model weights**
   - Place `sam_vit_h_4b8939.pth` in the repo root.
   - For SAM2, download a config + checkpoint pair (e.g. `sam2_hiera_large.yaml`, `sam2_hiera_large.pt`). Keep absolute paths handy.
5. **Configure the backend**
   ```bash
   cp .env.example .env
   ```
   Update `.env`:
   ```bash
   LOGREG_PATH=./my_logreg_model.pkl
   LABELMAP_PATH=./my_label_list.pkl
   CLIP_MODEL_NAME=ViT-B/32
   SAM_VARIANT=sam1                # or sam2
   SAM_CHECKPOINT_PATH=./sam_vit_h_4b8939.pth
   SAM2_CONFIG_PATH=/abs/path/to/sam2_config.yaml
   SAM2_CHECKPOINT_PATH=/abs/path/to/sam2_weights.pt
   ENABLE_METRICS=true             # optional Prometheus
   ```
   You can also point `CLIP_EMBED_CACHE` to customise where training caches embeddings (`./uploads/clip_embeddings` by default).
6. **Run the API**
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   Watch the logs for confirmations that CLIP, SAM, and the logistic regression model loaded correctly.
7. **Open the UI** – load `ybat-master/ybat.html` (locally renamed “Tator 🥔”) in your browser.

## Using the UI
### Label Images Tab
- Load images via the folder picker; per-image CLIP/SAM helpers live in the left rail.
- Toggle **Preload SAM** to stream the next image into memory; the side progress bar shows status and cancels stale tasks when you move to another image.
- Auto class, SAM box/point modes, and multi-point masks share a top progress indicator and support keyboard shortcuts documented in the panel footer.

### Train CLIP Tab
1. Choose **Image folder** and **Label folder** via native directory pickers. Only files matching YOLO expectations are enumerated.
2. (Optional) Provide a labelmap so class ordering matches the labeling tab.
3. Configure solver, class weights, max iterations, batch size, convergence tolerance, and hard-example mining (with adjustable weights/thresholds) plus **Cache & reuse embeddings** (enabled by default).
4. Select an output directory; training writes `{model,labelmap,meta}.pkl` plus JSON metrics.
5. Click **Start Training**. Progress logs stream live, including per-iteration convergence and per-class precision/recall/F1. Completed runs appear in the summary panel with download links.

Cached embeddings live under `uploads/clip_embeddings/<signature>/` and are keyed by dataset paths + CLIP backbone, independent of batch size. Toggling cache reuse will hit the store when inputs match.

### CLIP Model Tab
- Activate a classifier by picking its `.pkl` artifacts or by selecting a completed training run; metadata auto-selects the correct CLIP backbone and labelmap.
- Guidance text explains backbone auto-detection when a `.meta.pkl` file accompanies the classifier.

## Command-Line Training
The UI shares its engine with `tools/train_clip_regression_from_YOLO.py`:
```bash
python tools/train_clip_regression_from_YOLO.py \
  --images_path ./images \
  --labels_path ./labels \
  --labelmap_path my_label_list.pkl \
  --model_output my_logreg_model.pkl \
  --labelmap_output my_label_list.pkl \
  --solver saga --max_iter 1000 --device_override cuda
```
Use `--resume-cache` to reuse embeddings and `--hard-example-mining` to emphasise frequently misclassified classes.

## Development & Testing
- Run unit tests: `pytest`
- Static checks: `ruff check .`, `black --check .`, `mypy .`
- See `AGENTS.md` for coding conventions, PR expectations, and manual verification steps.

## Troubleshooting
- **Torch install errors** – install the wheel that matches your platform (`pip install torch==<version>+cu118 ...`).
- **SAM weights missing** – confirm paths in `.env`. SAM2 requires both config and checkpoint.
- **Large datasets** – enable caching (default) to avoid recomputing embeddings; caches are safe to prune manually.
- **Prometheus scraper fails** – ensure `/metrics` is enabled and FastAPI is reachable; the endpoint now serves plaintext output compatible with Prometheus.

## Credits
Built on top of [YBAT](https://github.com/drainingsun/ybat), [OpenAI CLIP](https://github.com/openai/CLIP), and Meta’s [SAM](https://github.com/facebookresearch/segment-anything) / [SAM2](https://github.com/facebookresearch/sam2). Novel code is released under the MIT License (see below). GIF assets in this README showcase the Auto Class workflows.


## LOP
1. **[planned]** SAM2 implementation is not properly tested yet - it's likely there are some issues to be cleaned up!
2. **[planned]** CLIP regression / training is in early stages - it works but it's likely we can develop some better default recipes
3. **[up for grabs]** We should add OBB support, it would be pretty simple to do in terms of UX and can really leverage SAM refinement
4. **[up for grabs]** Tracking / video sequence-annotation would be a cool longer-term objective. 
5. **[planned]** Docs should be improved, especially around explaining how to run the backend on a remore GPU-enabled server for bigger labeling jobs.
6. **[up for grabs]** Clean multi-user support would be nice in the future, using a single backend with some UX / UI to deconflict and distribute work packages.
7. **[planned]** Faster, faster! Everything should be made faster to keep the UX enjoyable.
8. **[up for grabs]** The logic of running the training from a remote server (transferring images in base64) is untested, and most likely buggy.
MRs welcome!

## License
Copyright (c) 2025 Aircortex.com — released under the MIT License. Third-party assets retain their original licenses.
