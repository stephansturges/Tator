# 🥔 Tator – Local CLIP + SAM Image Annotation Toolkit

Tator is a single-machine annotation workflow that pairs a clean, fast, simple, web-based frontend with a FastAPI backend to deliver _fast_ bounding-box annotation for images as well as some cool optional automation like class suggestions powered by CLIP and bbox cleanup / auto-suggestion using Segment Anything (SAM). The UI now bundles labeling, CLIP training, and model management in one place so you can iterate on datasets without leaving the browser.

## Lightning-Fast Labeling Modes

### Auto Class Corrector
Drop any tentative label and let CLIP clean it up instantly. Tator double-checks every box and snaps it to the class with the highest confidence so you can move through image stacks at warp speed.

![bbox_clip_correction](https://github.com/user-attachments/assets/b339541a-e60c-4091-b5bb-2ff105cf0bc6)


### Auto Box Refinement
Rough sketches are enough—SAM reshapes your loose bounding boxes into pixel-perfect rectangles while CLIP verifies the class. It feels like spell-check for geometry.

![bbox_sam_refiner](https://github.com/user-attachments/assets/04f678c4-5520-489e-ac06-aa34df0f60ce)


### One-Click Point-to-Box
Tap once anywhere on the object and SAM conjures a tight box while CLIP names it. Perfect for those “I just need this labeled now” marathons.

![clip_and_SAM](https://github.com/user-attachments/assets/746c12ba-3241-4fa4-8286-d06891cb54ca)

### Multi-Point Magic
When objects are tricky, sprinkle a few positive/negative points and let SAM sculpt the exact mask. Queue up new selections immediately—no waiting for the previous mask to land.

![multipoint_sam_and_clip](https://github.com/user-attachments/assets/d82e2f49-cc20-4927-b941-05cfe344817d)


### SAM Preload Boost
Enable preloading to keep the next image warmed up inside SAM. You’ll see progress ticks in the side rail and enjoy near-zero latency when you start pointing.


## Key Features
- **One-click assists** – auto class, SAM box/point refinements, and multi-point prompts with live progress indicators.
- **SAM 1 & SAM 2** – switch backends at runtime, optionally preload images into SAM to minimise round-trips.
- **Embedded CLIP trainer** – start training jobs from the UI, watch convergence metrics, and reuse cached embeddings across runs.
- **Model switcher** – activate new CLIP + regression pairs without restarting the server; metadata keeps backbone/labelmap in sync.
- **Predictor budget control** – dial the number of warm SAM predictors (1–3) and monitor their RAM usage so the UI can stay snappy on machines with more headroom.
- **One-click SAM bbox tweak** – press `X` while a bbox is selected to resubmit it through SAM (and CLIP if enabled) for a quick cleanup; double-tap `X` to fan the tweak out to the entire class.
- **Qwen 2.5 prompts** – zero-shot prompts spawn new boxes for the currently selected class; choose raw bounding boxes, have Qwen place clicks for SAM, or let it emit bounding boxes that immediately flow through SAM for cleanup. Prompt templates are editable in the dedicated Qwen Config tab so you can tailor instructions per project.
- **Live request queue** – a small corner overlay lists every in-flight SAM preload/activation/tweak so you always know what the backend is working on.
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
   QWEN_MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
   QWEN_DEVICE=auto                # cuda, cpu, mps, or auto
   QWEN_MAX_NEW_TOKENS=768         # clamp generation length if needed
   ```
   You can also point `CLIP_EMBED_CACHE` to customise where training caches embeddings (`./uploads/clip_embeddings` by default).
   Qwen loads lazily the first time you hit the "Use Qwen" button in the UI. Override `QWEN_MODEL_NAME` to point at a different Hugging Face repo or set `QWEN_DEVICE` to `cuda:1`, `cpu`, etc., if you need to pin the workload to a specific accelerator.
6. **Run the API**
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   Watch the logs for confirmations that CLIP, SAM, and the logistic regression model loaded correctly.
7. **Open the UI** – load `ybat-master/ybat.html` (locally renamed “Tator 🥔”) in your browser.
8. **Point the UI at a remote backend (optional)** – open the new **Backend** tab and enter the FastAPI base URL (e.g. `http://localhost:8000` when tunnelling over SSH). The value is saved locally so the browser reconnects automatically next time.

### Getting Started (First Project)
1. **Create a workspace folder** on your laptop with subfolders `images/` and `labels/` (YOLO-format `.txt`).
2. **Collect a labelmap:** create `my_label_list.txt` with one class per line or export it from YOLO training runs.
3. **Download model weights:**
   - Place `sam_vit_h_4b8939.pth` in the repo root.
   - (Optional) add SAM2 configs/checkpoints and point them via `.env`.
   - CLIP weights are auto-downloaded by `pip install -r requirements.txt` the first time.
4. **Install dependencies & copy `.env`:** see steps above (`python3 -m venv`, `pip install -r requirements.txt`, `cp .env.example .env`).
5. **Start the backend** (`python -m uvicorn app:app --host 0.0.0.0 --port 8000`).
6. **Open `ybat-master/ybat.html`:**
   - Click **Choose Images…** and select your local `images/` folder (browser uploads as needed).
   - Load classes via **Load Classes…** with `my_label_list.txt`.
   - Import existing YOLO boxes via **Import Bboxes…** — you can point it at a folder of `.txt` files or drop a `.zip` containing them.
   - Enable **SAM Mode** and/or **Auto Class** and start annotating.
7. **Training loop:** use the Train CLIP tab to train on the same `images/` + `labels/` folders, then activate the resulting `.pkl` via the CLIP Model tab.


### Running the Backend on a Remote GPU Host
You can keep the UI/data on your laptop and push all SAM/CLIP heavy lifting to a remote machine:

1. **Prepare the remote host** (GPU recommended):
   ```bash
   ssh user@gpu-host
   git clone https://github.com/aircortex/tator.git && cd tator
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```
   Keep this `uvicorn` process running (tmux/screen systemd, etc.).

2. **Tunnel the API locally** from your laptop:
   ```bash
   ssh -L 8000:127.0.0.1:8000 user@gpu-host
   ```
   As long as the tunnel is open, `http://localhost:8000` points to the GPU server.

3. **Open the UI on your laptop** (`ybat-master/ybat.html`). The default `API_ROOT = http://localhost:8000` already targets the tunnel, so the browser uploads images/bboxes over SSH and the remote SAM/CLIP predictors do the work.

#### Multi-user tips
- Run one FastAPI instance per user on unique ports (`uvicorn app:app --port 8010`, `8011`, …).
- Each user tunnels their assigned port (`ssh -L 8010:127.0.0.1:8010 user@gpu-host`).
- Optionally front the instances with an authenticated reverse proxy (nginx/traefik) so users hit different URLs without managing SSH tunnels.
- The UI is stateless: as long as `API_ROOT` points to the user’s tunnel/URL, their browser uploads images as base64 and never touches the remote filesystem.

## Using the UI
### Label Images Tab
- Load images via the folder picker; per-image CLIP/SAM helpers live in the left rail.
- Toggle **Preload SAM** to stream the next image into memory; the side progress bar shows status and cancels stale tasks when you move to another image.
- The **task queue overlay** in the lower-left corner lists every pending SAM preload/activation/tweak so you always know what work is queued up.
- The **Qwen 2.5 Assist** card now focuses on the object list: drop keywords such as `car, bus, kiosk` and the template (configured in the Qwen Config tab) builds the full prompt. Choose whether the detections stay as raw boxes, go through the new “Bounding boxes → SAM cleanup” mode, or emit SAM-ready points. Expand the advanced overrides to supply a one-off custom prompt, tweak the image-type description, or add extra context for tricky scenes.
- Press **`X`** with a bbox selected and SAM/CLIP will refine it in place; double-tap `X` to batch-tweak the entire class (the tweak always targets whichever class is currently selected in the sidebar). *(GIF placeholder)*
- Import YOLO `.txt` folders or zipped annotation bundles via the dedicated buttons—the app now streams bboxes even while images are still ingesting.
- Auto class, SAM box/point modes, and multi-point masks share a top progress indicator and support keyboard shortcuts documented in the panel footer.

#### Keyboard Shortcuts
- `X` – press while a bbox is selected to trigger the one-click SAM tweak.
- `X` `X` – double tap to batch-tweak every bbox of the current class.
- `A` – toggle Auto Class.
- `S` – toggle SAM Mode.
- `D` – toggle SAM Point Mode.
- `M` – toggle SAM Multi-Point Mode.
- `F` / `G` – add positive / negative multi-point markers.
- `Enter` – submit multi-point selection.
- Arrow keys – move through images/classes; `Q` removes the most recent bbox.

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

### Qwen Config Tab
- Edit the base prompts that power the Qwen Assist workflows. Each template must include `{image_type}`, `{items}`, and `{extra_context}` placeholders; the labeling tab fills them with the per-image inputs.
- Set sensible defaults for the type of imagery you work with (e.g. “overhead drone image”) and add boilerplate context (such as “Respond with JSON only”).
- Save changes, restore the defaults, or reload the live configuration at any time—handy when you run multiple backends with different prompt needs.
- The labeling tab exposes per-image overrides (custom prompt, image-type tweak, extra context) that layer on top of whatever defaults you set here.

### Predictors Tab
- Choose how many SAM predictors stay resident (current + optional next/previous) so you can preload in whichever direction you travel.
- See live stats for active/loaded slots, predictor RAM consumption, total FastAPI RAM usage, and free system memory. Values refresh automatically every few seconds while the tab is open.
- The Label Images tab respects this budget immediately: with 1 predictor only the current image stays hot, with 2 the “next” image preloads, and with 3 you also keep the previous image ready for instant backtracking.

### Backend Tab
- Configure the base URL that the UI uses for all API calls. Enter `http://host:port` (tunnels supported) and click **Save**; the setting persists in `localStorage` so it survives reloads.
- Use **Test Connection** to ping `/sam_slots` on the target server and confirm it’s reachable before switching tabs.
- Handy when you run the FastAPI backend on a remote GPU box and access it over SSH port forwarding.

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


## 2025-11-09 – Task Queue & Batch Tweaks

- Added a persistent task queue overlay that surfaces every pending SAM preload/activation/tweak so you can see exactly what the backend is chewing on.
- Double-tapping `X` now opens a batch-tweak prompt that runs the SAM cleanup across every bbox of the current class.
- Image ingestion + bbox imports now run concurrently; new progress toasts show when large batches are still being staged, and YOLO `.zip` bundles are supported alongside raw folders.
- Tweaks wait for any in-flight preload instead of forcing a brand-new predictor, eliminating the “stuck” state when hammering `X` on freshly loaded images.

## 2025-11-08 – Multi-Predictor Controller
- Unified the FastAPI backend so it always runs the multi-predictor SAM workflow with a configurable budget (1–3 slots) and exposes `/predictor_settings` for automation.
- Added a Predictors tab in the UI to adjust the budget, monitor slot counts, and watch RAM usage without leaving the browser.
- Taught the labeling tab to respect the budget automatically: the current image is always pinned, the “next” slot activates once you allow ≥2 predictors, and the “previous” slot joins in at ≥3, all while reusing in-flight preloads when you change images.
- Introduced One-click SAM bbox tweak so pressing `X` with a bbox selected resubmits it to SAM/CLIP for cleanup without redrawing.


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
