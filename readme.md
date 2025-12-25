/!\ Under active development. A lot of things are somewhat broken, especially in the dataset management side. SAM3 recipe "mining" as well as SAM3 training are both very much WIP. They work, but the methods are not optimized yet so don't take current performance as an indicator of anything. /!\


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
- **SAM3 Agent Mining** – mine portable, multi-step SAM3 recipes (ZIP) that combine text prompting + CLIP filtering (positive/negative crops) and optional pretrained CLIP heads.
- **Embedded CLIP trainer** – start training jobs from the UI, watch convergence metrics, and reuse cached embeddings across runs.
- **Model switcher** – activate new CLIP + regression pairs without restarting the server; metadata keeps backbone/labelmap in sync.
- **Predictor budget control** – dial the number of warm SAM predictors (1–3) and monitor their RAM usage so the UI can stay snappy on machines with more headroom.
- **One-click SAM bbox tweak** – press `X` while a bbox is selected to resubmit it through SAM (and CLIP if enabled) for a quick cleanup; double-tap `X` to fan the tweak out to the entire class.
- **Qwen 2.5 prompts** – zero-shot prompts spawn new boxes for the currently selected class; choose raw bounding boxes, have Qwen place clicks for SAM, or let it emit bounding boxes that immediately flow through SAM for cleanup. The active model (selected on the Qwen Models tab) always supplies the system prompt and defaults so inference matches training.
- **Live request queue** – a small corner overlay lists every in-flight SAM preload/activation/tweak so you always know what the backend is working on.
- **Prometheus metrics** – enable `/metrics` via `.env` for operational visibility.

## Repository Layout
- `app/`, `localinferenceapi.py` – FastAPI app, SAM/CLIP orchestration, training endpoints.
- `ybat-master/` – browser UI (`ybat.html`, CSS/JS, assets).
- `tools/` – reusable training helpers and CLI scripts.
- `sam3/` – optional upstream SAM3 checkout (needed for SAM3 training; inference can use downloaded checkpoints).
- `tests/` – unit tests.
- `uploads/` – runtime artifacts (CLIP embedding cache, Agent Mining jobs/recipes/cascades, Qwen runs, exports).

## How Tator Works (Architecture + Code Paths)

### High-level architecture
```text
┌───────────────────────────────────────────────────────────────────────────┐
│ Browser UI (ybat-master/ybat.html + ybat.js)                               │
│  - Label Images: manual boxes + assists (CLIP, SAM, SAM3 text)             │
│  - Train CLIP: dataset upload + logistic-regression head training          │
│  - Agent Mining: recipe search (mining) + save/export/import recipes       │
│  - Recipe cascades: chain recipes + de-dupe controls + save/export/import  │
│  - Qwen: (optional) train/activate/infer                                  │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │ HTTP/JSON (API_ROOT)
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ FastAPI backend (app/ + localinferenceapi.py)                              │
│  - Model runtime: CLIP, SAM1/2/3, (optional) Qwen                          │
│  - Training jobs: CLIP, Qwen, SAM3                                         │
│  - Agent Mining: jobs/results + apply single recipe + apply recipe cascade │
│  - Portability: recipe/cascade ZIP export + import                         │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │ reads/writes
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ On-disk state (uploads/)                                                  │
│  - clip_embeddings/   cached CLIP features used by training + mining       │
│  - agent_mining/      jobs/, cache/, recipes/, cascades/                   │
│  - qwen_runs/         datasets/, checkpoints/, metadata.json               │
└───────────────────────────────────────────────────────────────────────────┘
```

### Codebase map (where to look)
```text
Tator/
├─ app/                  FastAPI app object (uvicorn imports this)
├─ localinferenceapi.py  Backend endpoints + orchestration (CLIP/SAM/Qwen/Agent Mining)
├─ ybat-master/          Frontend UI (static HTML/JS/CSS)
├─ tools/                CLI helpers (e.g., CLIP training script)
├─ tests/                Unit tests for key flows
└─ uploads/              Runtime artifacts + caches (git-ignored)
```

### Major code paths (end-to-end)
- **CLIP / DINOv3 heads (recommended first)**: `Train CLIP` tab → `/clip/train` job → activate via `/clip/active_model` → used by Auto Class, CLIP verification (CLIP only), and as a **pretrained head** for Agent Mining / cascade scoring.
- **SAM assists (Label Images tab)**: UI actions → SAM endpoints (SAM1/2/3 depending on `SAM_VARIANT`) → detections written into the current image session.
- **SAM3 text prompt**: `SAM3 Text Prompt` panel → SAM3 processor runtime → returns boxes/polygons via the same `QwenDetection` response model used elsewhere.
- **Agent Mining (recipe search)**: Agent Mining tab → `/agent_mining/jobs` (start/poll/cancel) → results include per-class recipes in either:
  - `sam3_greedy`: prompt bank + optional crop bank (positives/negatives) + optional embedded pretrained CLIP head
  - `sam3_steps` (`schema_version=2`): explicit multi-step prompt chain (step list), plus optional embedded pretrained CLIP head
  Save/export as portable ZIP via `/agent_mining/recipes`.
- **Apply one recipe**: UI → `/agent_mining/apply_image` → `_apply_agent_recipe_to_image()`:
  - `sam3_steps`: run each step (seed → diverse seed selection → SAM3 visual expand → final CLIP filter + IoU de-dupe)
  - `sam3_greedy`: prompt bank → (optional) crop-bank CLIP filter → SAM3 expand → (optional) CLIP head gate → IoU de-dupe
- **Apply a recipe cascade**: UI → `/agent_mining/apply_image_chain` → run multiple recipes → per-class de-dupe → optional cross-class de-dupe (by group or global) with optional CLIP-head-based confidence → detections returned to UI.

## Prerequisites
- Python 3.10 or newer (3.11+ recommended).
- Optional GPU with CUDA for faster CLIP/SAM inference.
- Model weights: `sam_vit_h_4b8939.pth` (SAM1). Optional SAM3 checkpoints/configs are supported; see `sam3integration.txt` for sample commands and Hugging Face IDs.

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
   - (Optional) set up SAM3 support (see below) so the new SAM3 text prompts have weights available.
5. **Configure the backend**
   ```bash
   cp .env.example .env
   ```
   Update `.env`:
   ```bash
   LOGREG_PATH=./my_logreg_model.pkl
   LABELMAP_PATH=./my_label_list.pkl
   CLIP_MODEL_NAME=ViT-B/32
   SAM_VARIANT=sam1
   SAM_CHECKPOINT_PATH=./sam_vit_h_4b8939.pth
   SAM3_MODEL_ID=facebook/sam3
   SAM3_PROCESSOR_ID=facebook/sam3
   SAM3_CHECKPOINT_PATH=
   SAM3_DEVICE=
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
   - CLIP weights are auto-downloaded by `pip install -r requirements.txt` the first time.
4. **Install dependencies & copy `.env`:** see steps above (`python3 -m venv`, `pip install -r requirements.txt`, `cp .env.example .env`).
5. **Start the backend** (`python -m uvicorn app:app --host 0.0.0.0 --port 8000`).
6. **Open `ybat-master/ybat.html`:**
   - Click **Choose Images…** and select your local `images/` folder (browser uploads as needed).
   - Load classes via **Load Classes…** with `my_label_list.txt`.
   - Import existing YOLO boxes via **Import Bboxes…** — you can point it at a folder of `.txt` files or drop a `.zip` containing them.
   - Enable **SAM Mode** and/or **Auto Class** and start annotating.
7. **Train CLIP (recommended first):** use the **Train CLIP** tab to train on the same `images/` + `labels/` folders, then activate the resulting `.pkl` via the **CLIP Model** tab. This is the foundation for Auto Class, fast verification, and the best results when searching/mining recipes. It can run quickly on CPU/MPS too (e.g. on an M1 MacBook) and works the same whether your backend is local or on a remote GPU host.

### Optional: Setting up SAM3
SAM3 support is optional but recommended if you plan to use the text-prompt workflow. Follow Meta’s instructions plus the notes below (summarised from `sam3integration.txt`):

1. **Request checkpoint access** — visit the [facebook/sam3](https://huggingface.co/facebook/sam3) page and request access. Hugging Face will email you once approved.
2. **Install the official SAM3 repo** — clone Meta’s implementation and install it in editable mode (this provides the `build_sam3_image_model` + processor used by our backend):
   ```bash
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .           # installs the sam3 package
   pip install einops         # SAM3 depends on einops; install it if not pulled automatically
   ```
3. **Authenticate with Hugging Face** — run
   ```bash
   hf auth login
   ```
   Generate a read token from your Hugging Face settings, paste it when prompted, and verify with `hf auth whoami`. This allows Transformers to download the gated checkpoints automatically.
4. **(Optional) Pin checkpoints manually** – if you want deterministic paths, call `huggingface_hub.hf_hub_download` (examples in `sam3integration.txt`) and set `SAM3_CHECKPOINT_PATH` / `SAM3_MODEL_ID` to the downloaded files.
5. **Run the API** — once authenticated, start the backend as usual. Selecting “SAM 3” in the UI enables both the point/bbox flows and the new text prompt panel.

### Segmentation Builder (bbox → polygons)
The **Segmentation Builder** tab clones an existing **bbox** dataset into a YOLO‑seg (polygon) dataset using SAM1 or SAM3. Originals stay untouched; output is named `<source>_seg` by default and tagged with `type: seg` so SAM3 training can auto-enable mask losses.

Flow:
- Pick any bbox dataset discovered under the unified **Datasets on disk** list (Qwen, SAM3, or registry roots).
- Choose **SAM variant** (SAM1 for classic masks, SAM3 for text/visual parity with recipe mining).
- Optional knobs (API payload, defaults shown): `mask_threshold=0.5`, `score_threshold=0.0`, `simplify_epsilon=30`, `min_size=0`, `max_results=1`.
- Click **Start build**. A job is queued; progress and logs stream live in the panel. Jobs survive page reload; polling resumes automatically.
- Each image is loaded once, all boxes are prompted in parallel across available GPUs (2 workers per device by default; override with env `SEG_BUILDER_WORKERS_PER_DEVICE`, `SEG_BUILDER_MAX_WORKERS`).
- For each bbox: best SAM mask → polygon (convex hull + RDP simplification). If mask is below the score threshold or unusable, we fall back to the original box as a 4‑point polygon.
- Output layout: `train/` and `val/` with `images/` (hard-linked or copied) and `labels/` in YOLO‑seg polygon format, plus `labelmap.txt` and `sam3_dataset.json`. COCO `_annotations.coco.json` is regenerated for both splits.
- **Screenshots coming soon** (build list + logs).

Error handling & requirements:
- Requires at least one class in metadata or `labelmap.txt` (builder fails fast otherwise).
- SciPy is optional; if missing, polygon fallback still works (bounding boxes).
- Progress is reported as images processed; logs include per-split counts and conversion notes.

#### Experimental: Training SAM3 (box-only)
> **Unstable:** this path is still changing. Training currently requires a checkout of the upstream SAM3 repo inside this project; expect sharp edges.

- **Clone upstream into Tator:** from the repo root  
  `git clone https://github.com/facebookresearch/sam3.git sam3`  
  then `pip install -e sam3 && pip install einops`.
- **GPU + CUDA Torch required:** ensure your `torch` build can see CUDA; CPU-only installs will fail.
- **Kick off training:** start `uvicorn` as usual, open the **Train SAM3** tab, pick a dataset (reuses cached Qwen datasets without re-upload), and click **Start**. Logs stream live and the loss line chart updates as batches finish.
- **Activate the checkpoint:** when a run completes, use **Activate checkpoint** in the same tab to swap the backend SAM3 model to your finetune.
- **Monkeypatch defaults ON:** we keep SAM3’s segmentation head weights for text prompting, but freeze/ignore segmentation-head + mask-FPN params during bbox-only training to avoid DDP unused-parameter crashes. The patch is applied automatically; disable it only if you truly train masks:
  ```bash
  SAM3_MONKEYPATCH=0 python -m uvicorn app:app --host 127.0.0.1 --port 8000
  ```
  Otherwise, launch `uvicorn` normally and the patch stays enabled.


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
- The **Qwen 2.5 Assist** card now focuses on the object list: drop keywords such as `car, bus, kiosk` and the template pulled from the active Qwen model builds the full prompt. Choose whether the detections stay as raw boxes, go through the new “Bounding boxes → SAM cleanup” mode, or emit SAM-ready points. Expand the advanced overrides to supply a one-off custom prompt, tweak the image-type description, or add extra context for tricky scenes.
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
- `Backspace` / `W` – delete the currently selected bbox (⌘+Delete on macOS still works too).

### Train CLIP Tab
1. Choose **Encoder type** (CLIP or DINOv3). CLIP enables text-based prompt prefiltering; DINOv3 is image-only and skips text prompts. If a DINOv3 model is gated, run `huggingface-cli login` (or set `HF_TOKEN`) and accept the license on Hugging Face.
2. DINOv3 heads **require the `.meta.pkl`** produced by training so the encoder type/model is known. Legacy heads without meta are treated as CLIP and cannot be activated as DINOv3.
3. Choose **Image folder** and **Label folder** via native directory pickers. Only files matching YOLO expectations are enumerated.
4. (Optional) Provide a labelmap so class ordering matches the labeling tab.
5. Configure solver, class weights, max iterations, batch size, convergence tolerance, and hard-example mining (with adjustable weights/thresholds) plus **Cache & reuse embeddings** (enabled by default).
6. Training outputs are written automatically to <code>uploads/classifiers/</code> (classifier + meta) and <code>uploads/labelmaps/</code> (labelmap). Output paths are fixed to keep model management consistent.
7. Click **Start Training**. Progress logs stream live, including per-iteration convergence and per-class precision/recall/F1. Completed runs appear in the summary panel with download links.

Cached embeddings live under `uploads/clip_embeddings/<signature>/` and are keyed by dataset paths + encoder model, independent of batch size. Toggling cache reuse will hit the store when inputs match.

#### Classifier Benchmarks (CLIP/DINOv3)
We run all classifier benchmarks on a **fixed group split** (20% by image, seed 42) so scores are comparable. Metrics below are **foreground-only** (exclude `__bg_*`) and come from the same cached embeddings used at training time. Full procedure lives in `classifier_testing_methodology.md`.

Recent DINOv3 MLP comparisons (qwen_dataset, 5 bg classes):

| Encoder | MLP sizes | Label smoothing | FG macro precision | FG macro recall | FG macro F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| dinov3-vits16 | 256 | 0.0 | 0.7635 | 0.8532 | 0.7991 | 0.8800 |
| dinov3-vits16 | 256 | 0.1 | 0.7348 | 0.8618 | 0.7789 | 0.8658 |
| dinov3-vitb16 | 512 | 0.0 | 0.7311 | 0.8423 | 0.7714 | 0.8603 |
| dinov3-vitb16 | 512 | 0.1 | 0.7484 | 0.8591 | 0.7892 | 0.8718 |
| dinov3-vitl16 | 768,384 | 0.0 | 0.7541 | 0.8465 | 0.7908 | 0.8732 |
| dinov3-vitl16 | 768,384 | 0.1 | 0.7991 | 0.8548 | 0.8226 | 0.8996 |

Takeaways:
- **Soft targets (label smoothing 0.1)** improved recall for all three sizes.
- Precision **improved for vitb16/vitl16** and dipped slightly for vits16.
- Best overall in this batch: **dinov3-vitl16 + MLP 768/384 + smoothing 0.1**.

### CLIP Class Predictor Settings Tab
- Activate a classifier by picking its `.pkl` artifacts or by selecting a completed training run; metadata auto-selects the correct encoder type/model and labelmap.
- Manage saved heads: **Refresh**, **Download zip** (classifier + meta + labelmap if found), or **Delete** directly from the list under <code>uploads/classifiers/</code>.
- Uploading a local `.pkl` stores it under <code>uploads/classifiers/</code> and adds it to the saved list (legacy heads without meta are treated as CLIP).
- CLIP heads use the CLIP backbone selector; DINOv3 heads ignore the CLIP backbone setting.
- Optional **Auto Class guard** can require a minimum confidence gap between the top two classes before accepting a suggestion.

### Train Qwen Tab
- Converts whatever images + YOLO boxes you loaded in the labeling tab into JSONL + image bundles automatically. The browser shuffles the set, builds an 80/20 train/val split, and streams the dataset via `/qwen/dataset/init` → `/qwen/dataset/chunk` → `/qwen/dataset/finalize`—no manual dataset paths required.
- Describe the imagery ("warehouse CCTV", "aerial rooftops", etc.), pick the base Hugging Face repo, and provide a single system prompt covering both bbox/point outputs. A “prompt noise” field (default 0.05) randomly drops a small percentage of characters so the adapters see slightly perturbed instructions every epoch.
- At training time each conversation randomly alternates between bbox vs. point outputs **and** between “all classes”, “single class”, and “subset of classes” instructions, so the adapters learn both broad sweeps and class-specific prompts without bloating the dataset on disk.
- Choose LoRA or QLoRA, tweak batch size/epochs/LR/adapter ranks, and hit **Start Training**. The backend unpacks the dataset under `uploads/qwen_runs/datasets/<run_name>`, launches the Lightning trainer, and streams progress/logs/checkpoints back to the UI.
- New knobs expose the image resize cap (default 1024&nbsp;px, applied to the longest side before Qwen sees the tensors) plus the per-sample detection limit (default 200 with the existing per-class balancer). Drop either value when you need to wrangle VRAM without editing code.
- Every run writes a `metadata.json` alongside the checkpoints with the system prompt, dataset context, and class list so you can reload the exact format later from the Qwen Models tab.
- Job cards show live progress, allow cancellation, and list the checkpoint folders that land under `uploads/qwen_runs/<run_name>/`. Completed runs stick around in the history list so you can re-download or re-run with tweaked prompts later.

### Qwen Models Tab
- Lists every custom Qwen fine-tune saved under `uploads/qwen_runs/`, plus the built-in base model. Each card shows the dataset description, class list, and stored system prompt so you always know how it was trained.
- Activate a model to make the Assist panel reuse its exact prompts and defaults; the items/context fields on the labeling tab auto-populate from the active metadata so inference matches training.
- Use the Train Qwen tab to produce new adapters, then switch between them here without touching the backend.

### SAM Predictors Tab
- Choose how many SAM predictors stay resident (current + optional next/previous) so you can preload in whichever direction you travel.
- See live stats for active/loaded slots, predictor RAM consumption, total FastAPI RAM usage, and free system memory. Values refresh automatically every few seconds while the tab is open.
- The Label Images tab respects this budget immediately: with 1 predictor only the current image stays hot, with 2 the “next” image preloads, and with 3 you also keep the previous image ready for instant backtracking.

### Backend Config Tab
- Configure the base URL that the UI uses for all API calls. Enter `http://host:port` (tunnels supported) and click **Save**; the setting persists in `localStorage` so it survives reloads.
- Use **Test Connection** to ping `/sam_slots` on the target server and confirm it’s reachable before switching tabs.
- Handy when you run the FastAPI backend on a remote GPU box and access it over SSH port forwarding.

### Agent Mining Tab (SAM3 Recipe Mining)
The **Agent Mining** tab mines **portable SAM3 “recipes”** for each class in a labeled dataset. A recipe is a **single ZIP** that includes:
- `recipe.json` (portable parameters + summary metrics + either a prompt bank (`sam3_greedy`) or an explicit step list (`sam3_steps`, schema v2))
- optional `crops/` (example cut-outs used for crop-bank CLIP filtering; not used in head-only recipes)
- optional `clip_head/` (a pretrained CLIP classifier head exported from the CLIP training tab)

*(Screenshot coming: Agent Mining tab overview + per-class recipe details.)*

Note: **CLIP prompt prefiltering is CLIP-only**. If you select a DINOv3 head, the prompt prefilter toggle is disabled and recipe mining skips text-embedding filters.

#### Recommended workflow (best results)
1. **Train CLIP first** on the dataset (Train CLIP tab). This gives you fast Auto Class while labeling and unlocks the most reliable “recipe search” because CLIP embeddings + (optional) pretrained CLIP heads become dataset-specific.
2. **Mine recipes** in Agent Mining using a small validation percentage first (smoke test), then scale up once the logs/results look sane.
3. **Apply + iterate** on a few representative images, then save/export the recipes you want to reuse elsewhere.

High-level flow (what it’s doing, in human terms):
1. **Split the dataset** into train/val (random seed is exposed so results are repeatable).
2. Build a **prompt bank** for each class (class name + optional GPT‑OSS suggestions + optional user-provided extra prompts).
3. Choose a filtering strategy:
   - **Pretrained CLIP head selected (recommended):** crop-bank settings are ignored and we filter using the classifier head instead.
   - **Crop-bank mode (no CLIP head):** we sample positive crops (and optional negatives) from the train split and use CLIP similarity as the filter.
4. Choose a recipe search mode:
   - **Greedy (fast):** evaluate using the full prompt bank (one “greedy” recipe per class).
   - **Multi-step (precision-first):** select a **small chain of prompt steps** per class and save it as a schema‑v2 step recipe (`mode=sam3_steps`).
     - Seed-stage: run **text-only** prompt detections on the val split and pick prompt steps using a greedy set-cover (maximize new GT coverage, tie-break on seed precision / fewer seed FPs).
     - Full-stage: for the selected steps, run a per-image pipeline: **seed → pick diverse seeds → SAM3 visual expand**.
     - Final-stage: if a pretrained CLIP head is selected, **sweep head thresholds** (`min_prob`, optional `margin`) and pick the best operating point for your chosen **Target precision**; these tuned thresholds get baked into the saved recipe ZIP.
   - **Beam search (slower):** explores more parameter combinations for greedy recipes.
5. Score each class recipe on the val split (coverage/precision/FPs) and show results. Click **Save recipe** to persist the portable ZIP on disk.

Important UI note: when you see **“CLIP filter off”** in Agent Mining results, that refers to **crop-bank CLIP similarity** (positive/negative crops). If you are using a pretrained CLIP head you should still see **“pretrained CLIP head …”** — that is the CLIP-based filtering being applied in that run.

#### Interpreting the scores
- **Coverage** = how many ground-truth objects were matched (recall): `matches / ground_truth`.
- **Precision** = `matches / (matches + false_positives)` on the val split.
- **FPs** = detections that do not match any ground-truth box at the job’s Eval IoU threshold.
- **Duplicates** = extra detections that overlap a ground-truth box that was already matched (double-labeling the same object).
- **Det rate** = fraction of val images that produced at least one detection.

Using it:
- Pick a converted dataset under **Datasets on disk**.
- Set `Val %` and `Split seed` (default seed 42). Leave **Reuse split** on if you want stable comparisons across runs.
- Configure crops/prompts/filters, then click **Start**. Watch the live log + progress bar.
- If you select a **Pretrained CLIP head**: leave **Auto-tune thresholds** on and set **Target precision** (e.g. 0.90–0.98). Higher = cleaner (fewer false positives) but lower recall.
- Review per-class results and click **Save recipe** for the classes you want.

Notes:
- IDs shown during mining are **dataset category ids** (COCO `category_id` when the dataset is COCO-format) and may not match your labelmap index/order.
- A recipe’s **CLIP head index** (shown as “head idx …”) is the index inside the classifier head’s class list; it is normal for this to differ from your labelmap index. Head matching is done by **class name** (after normalization), not by numeric ID.
- If a recipe class name can’t be found inside the embedded CLIP head, head-based filtering is skipped for that recipe (expect noisier results). Use a cascade **Output class override** or re-mine with consistent class names.

#### Apply a single recipe
- **Agent Mining tab (debugging):** select a saved recipe and apply it to a specific dataset image id to sanity-check what the recipe is doing.
- **Label Images tab → SAM3/Recipes panel:** import a recipe ZIP and click **Apply recipe to image** to write detections into the current image. If the recipe class doesn’t exist in the current labelmap, enable **Output class override** and pick the destination class.

#### Chain recipes with a cascade (Label Images tab)
Recipe cascades let you run multiple saved recipes in order and then clean up overlaps in a final de-dupe pass.

1. In **Label Images** → **SAM3/Recipes**, click **+ Add recipe step** to build a list of steps (reorder as needed).
2. Per step, choose the recipe and (optionally) set:
   - **Output class override** (re-label outputs to a different class)
   - **Dedupe group** (group steps that should de-dupe against each other)
   - **Participate in cross-class de-dupe** (turn off for cases like person-on-bike where overlap is expected)
   - **Extra CLIP min prob / Extra CLIP margin** (optional): add an extra CLIP-head confidence check for that step when applying the cascade. This is cumulative with the recipe’s baked-in CLIP thresholds (effective = `max(recipe, extra)`), and only applies to recipes that include an embedded pretrained CLIP head.
3. Configure de-dupe:
   - **Per-class de-dupe IoU**: removes duplicates within the same output class.
   - **Cross-class de-dupe** (optional): removes duplicates across different classes, either **within dedupe groups** or **globally**.
   - **Confidence (de-dupe)**: choose how “winner” detections are picked (SAM score, or CLIP-head-based confidence).
   - **CLIP head source recipe**: when using a CLIP-head confidence mode, pick which saved recipe provides the embedded head.
4. Click **Apply cascade to image** to write the final merged detections into the current image.

*(Screenshot coming: cascade editor + de-dupe settings.)*

#### Save / reuse cascades
- Use **Save cascade preset** to persist the cascade configuration on the backend.
- Use **Download preset (zip)** to export a portable zip that bundles `cascade.json` plus all referenced recipe zips.
- Use **Import cascade (zip)** to bring the whole bundle onto another machine/backend.

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
- Project notes live in `SESSION.md` and `LABEL_SCHEMA_REFACTOR_BRIEF.md`.

## Troubleshooting
- **Torch install errors** – install the wheel that matches your platform (`pip install torch==<version>+cu118 ...`).
- **SAM weights missing** – confirm paths in `.env`. Ensure `SAM_CHECKPOINT_PATH` points at the desired `.pth`.
- **Large datasets** – enable caching (default) to avoid recomputing embeddings; caches are safe to prune manually.
- **Prometheus scraper fails** – ensure `/metrics` is enabled and FastAPI is reachable; the endpoint now serves plaintext output compatible with Prometheus.

## Credits
Built on top of [YBAT](https://github.com/drainingsun/ybat), [OpenAI CLIP](https://github.com/openai/CLIP), and Meta’s [SAM](https://github.com/facebookresearch/segment-anything). Novel code is released under the MIT License (see below). GIF assets in this README showcase the Auto Class workflows.

## 2025-12-25 – DINOv3 MLP Benchmarks + Soft Targets
- Benchmarked **DINOv3 + MLP** heads (small/base/large) with and without **label smoothing** and logged results to the comparison tables.
- Added a concise methodology (`classifier_testing_methodology.md`) so future classifier runs are evaluated on the same split + metrics.
- Updated the Train CLIP section with a quick summary table and the precision/recall tradeoffs from soft targets.

## 2025-12-20 – DINOv3 Heads + CLIP Model Management
- Added **DINOv3 encoder support** for classifier heads (image-only alternative to CLIP). DINOv3 heads require `.meta.pkl` so the encoder model is known.
- Consolidated **CLIP model management**: refresh/download/delete saved heads and labelmaps, upload local `.pkl` heads, and activate models from a single UI panel.
- Hardened classifier loading: missing meta now falls back to CLIP with backbone inference from embedding size (512 → ViT‑B/32, 768 → ViT‑L/14); non‑CLIP heads require meta.
- UI status updates now surface encoder readiness/errors so you can see when a head can’t load.

## 2025-12-18 – Recipe ZIP Import Fix
- Rebuilt Agent Mining’s “Multi-step” mode so it produces **schema‑v2 step recipes** (`mode=sam3_steps`, `schema_version=2`) instead of a legacy/implicit structure.
  - Mining now explicitly selects a **chain of prompt steps** per class (up to `Max recipe steps`) and saves them as `recipe.steps[]`.
  - Each step runs a consistent per-image pipeline: **seed → diverse seed selection → SAM3 visual expand → final filter + de-dupe**.
  - When a pretrained CLIP head is used, we **auto-tune the head thresholds** (probability + optional margin) to hit the user’s **Target precision** and bake that into the recipe ZIP.
- Fixed recipe ZIP import so it **preserves the saved top-level `params` block** (tuned thresholds/knobs) instead of dropping it. This also applies when importing cascade ZIP bundles (they import recipe ZIPs internally).
- Hardened step recipe saving so the on-disk recipe JSON is unambiguous (`schema_version=2`, `mode=sam3_steps`) even if the incoming payload relied on implicit classification.

## 2025-12-17 – Recipe Cascades + CLIP Head Robustness
- Added **recipe cascades** in the Label Images tab: chain multiple recipes, optionally re-label outputs per step, and merge results with configurable per-class + cross-class IoU de-dupe.
- Added **dedupe groups** + per-step opt-out for cross-class de-dupe (useful when overlap is expected, e.g. person-on-bike).
- Added **cascade presets** with backend save/load plus portable ZIP export/import (bundle includes cascade + all referenced recipes).
- Added **per-step CLIP head threshold overrides** when applying cascades (cumulative with baked-in recipe thresholds) so you can tighten filtering at inference time without re-mining.
- Improved CLIP-head reliability + debugging: infer head proba mode when metadata is missing, warn when a recipe class can’t be found in the head, and clarify class-id vs labelmap-index vs head-class-index in the UI.
- Agent Mining: CLIP head threshold tuning is now **precision-first**, with a simple **Target precision** control and per-class auto-tuning of `min_prob` + margin.

## 2025-12-16 – Agent Mining Recipes (SAM3) + Pretrained CLIP Head
- Added a full **Agent Mining** UI to mine per-class SAM3 recipes and manage saved recipes (list/import/export/delete).
- Recipes are now **portable ZIPs** that bundle `recipe.json` + exemplar crops and optional CLIP head artifacts, so they can be copied to another machine without external paths.
- New `/agent_mining/apply_image` (and `/agent_mining/apply_image_chain` for cascades) endpoints run the full recipe pipeline (text seeds → CLIP filter → SAM3 visual expansion → CLIP/IoU dedupe) and return boxes/polygons.
- Added **output class override** in the recipe apply UI so you can apply a recipe to any labelmap (with warnings for mismatches).
- Added **Pretrained CLIP head** option (exported from CLIP training artifacts) for additional filtering; mining can auto-tune per-class head thresholds on the validation split (without re-running SAM3 multiple times).
- Improved Agent Mining **logging + progress reporting**, plus a **cache size** view and **purge cache** button to reclaim disk.
- Hardened recipe portability: sanitize crop paths, embed crops under `crops/`, and validate recipe schema on load/import.


## 2025-11-10 – Qwen Assist & Backend Controls

- Landed first-class Qwen 2.5 support: the backend now mirrors the PyImageSearch zero-shot recipe (chat templates + `process_vision_info`) and exposes `/qwen/infer` so the UI can request raw boxes, bbox→SAM cleanups, or SAM-ready click points.
- Added a **Qwen Config** tab plus an Assist card in the labeling view. You can edit the base `{image_type}/{items}/{extra_context}` templates, override prompts per image, pick output modes, and stream responses directly into YOLO boxes without touching scripts.
- Introduced a **Backend** tab so the browser can point at any FastAPI root (local or tunneled) without code edits; the README now documents the remote GPU workflow end-to-end, including the new Qwen env vars.

## 2025-11-12 – Qwen Model Manager

- The old Qwen Config editor was replaced with a **Qwen Models** tab that lists every fine-tuned run plus the base model. Each card surfaces the saved system prompt, dataset context, and class list so you always know how the adapters were trained.
- Active model metadata now drives the Assist panel: the system prompt is injected automatically at inference time and the image/context fields pick up the defaults from the selected run.
- Backend training writes `metadata.json` next to each checkpoint and new `/qwen/models` + `/qwen/models/activate` endpoints let you hot-swap adapters without touching the CLI.
- Expanded Quick Start docs and the shortcut list (hello `W` delete hotkey) so every UI change ships with matching guidance.


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
1. **[future]** Disentangle from Meta’s SAM3 repo with an in-repo trainer; postponed until the current SAM3 path is stable.
2. **[planned]** CLIP regression / training works but needs better default recipes and tuning.
3. **[up for grabs]** Add oriented bounding-box support to better leverage SAM refinement.
4. **[up for grabs]** Tracking / video sequence-annotation remains a longer-term objective.
5. **[planned]** Improve docs, especially for running the backend on remote GPU hosts and clarifying SAM3 training/activation.
6. **[up for grabs]** Clean multi-user support on a shared backend with UX to deconflict work packages.
7. **[planned]** Performance: keep pushing latency/throughput down for a smoother UX.
8. **[up for grabs]** Remote training via base64 transfer is still untested and likely buggy.
9. **[future]** Optional SAM3 hard-example mining toggle (replay top-loss batches after burn-in), off by default to keep the trainer simple.
MRs welcome!

## License
Copyright (c) 2025 Aircortex.com — released under the MIT License. Third-party assets retain their original licenses.
