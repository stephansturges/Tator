/!\ Under active development. Dataset management + training flows are active areas of work; expect periodic changes. /!\


# 🥔 Tator – Local CLIP + SAM Image Annotation Toolkit

Tator is a single-machine annotation workflow that pairs a clean, fast, simple, web-based frontend with a FastAPI backend to deliver fast bounding‑box labeling, light automation (CLIP class suggestions + SAM geometry fixes), and a deterministic **deep prelabeling** pipeline that combines multiple detectors + SAM3 and filters them through a calibration model. The UI bundles labeling, training, dataset management, and prelabeling in one place so you can iterate without leaving the browser.

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
- **Qwen 3 prompts** – zero-shot prompts spawn new boxes for the currently selected class; choose raw bounding boxes, have Qwen place clicks for SAM, or let it emit bounding boxes that immediately flow through SAM for cleanup. The active model (selected on the Qwen Models tab) always supplies the system prompt and defaults so inference matches training.
- **Qwen 3 captioning** – generate long-form captions with label hints and optional reasoning models, then save them into a `text_labels/` folder alongside YOLO exports (use the Label Images panel).
- **Deep prelabeling** – deterministic prepass (detectors + SAM3 + dedupe) with optional glossary expansion + **calibration (XGBoost)** to filter candidates.
- **Live request queue** – a small corner overlay lists every in-flight SAM preload/activation/tweak so you always know what the backend is working on.
- **YOLOv8 training** – launch detect/segment runs from the UI, track progress, and keep only `best.pt` + metrics for easy sharing.
- **RF-DETR training** – launch detect/segment runs from the UI, track progress, and keep best checkpoints + metrics for easy sharing.
- **Segmentation safeguards** – segmentation training only runs on polygon datasets; Dataset Management shows YOLO‑SEG / COCO‑SEG readiness.
- **Prometheus metrics** – enable `/metrics` via `.env` for operational visibility.

## Repository Layout
- `app/`, `localinferenceapi.py` – FastAPI app, SAM/CLIP orchestration, training endpoints.
- `ybat-master/` – browser UI (`ybat.html`, CSS/JS, assets).
- `tools/` – reusable training helpers and CLI scripts.
- `sam3/` – optional upstream SAM3 checkout (needed for SAM3 training; inference can use downloaded checkpoints).
- `tests/` – unit tests.
- `uploads/` – runtime artifacts (CLIP embedding cache, Agent Mining jobs/recipes/cascades, Qwen runs, calibration jobs + caches, glossaries, exports).

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
│  - Training jobs: CLIP, Qwen, SAM3, YOLOv8, RF-DETR                         │
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
│  - calibration_jobs/  job state + logs for prepass calibration             │
│  - calibration_cache/ cached prepass/features/labels for calibration       │
│  - glossaries/        saved glossary library (glossaries.json)             │
│  - yolo_runs/         best.pt, metrics.json, results.csv, run.json         │
│  - rfdetr_runs/       checkpoints, metrics, results.json, run.json         │
└───────────────────────────────────────────────────────────────────────────┘
```

### Deep prelabeling (prepass + calibration)
- Open the **Deep prelabeling** panel to configure detectors + SAM3.
- The prepass is deterministic: detectors (YOLO/RF‑DETR), SAM3 text + similarity, and IoU dedupe.
- Optional glossary expansion uses Qwen to propose extra SAM3 terms per class.
- Calibration (XGBoost) filters the candidate pool to maximize F1 while enforcing a recall floor.
- Endpoint: `POST /qwen/prepass` (returns detections + a compact trace of prepass steps).
- We previously experimented with an agentic annotation loop. It was removed in favor of this deterministic prepass + calibration stack, which is faster, more stable, and easier to tune. Qwen is now only used for glossary expansion and captioning.

### Prepass architecture (detectors + SAM3 + dedupe)
The prepass is a deterministic, multi‑stage detector stack that builds a high‑recall candidate pool before calibration.

```text
Full image
   │
   ├─ Detectors (YOLO, RF‑DETR)
   │    ├─ full‑frame pass
   │    └─ SAHI windowed pass (slice + merge)
   │
   ├─ SAM3 text (glossary terms + optional Qwen expansion)
   │
   ├─ Dedupe A (IoU merge) + optional cleanup
   │    └─ classifier cleanup only if prepass_keep_all=false
   │
   ├─ SAM3 similarity (global full‑frame)
   │    └─ optional windowed similarity extension
   │
   └─ Dedupe B (IoU merge) + optional cleanup
        └─ final prepass candidate set (with provenance)
```

Key notes:
- **Full‑frame + SAHI**: every detector runs twice (full‑frame and SAHI), then both streams are merged in the same dedupe pass.
- **SAM3 text**: uses the dataset glossary by default; Qwen can optionally add extra terms (capped per class).
- **Similarity**: always runs global similarity; windowed similarity is an opt‑in extension.
- **Dedupe**: run twice (after detectors + text, and after similarity) to stabilize cluster assignment.
- **Calibration**: the calibration prepass uses `prepass_keep_all=true` and `prepass_caption=false` so the MLP sees the full candidate pool with no classifier gating.

### YOLO‑first dataset management
Tator treats YOLO as canonical. Every dataset should have:
- `labelmap.txt`
- `train/images/`, `train/labels/`
- optional `val/images/`, `val/labels/`

If a dataset exists only as COCO, the backend **auto‑converts COCO → YOLO** during dataset listing so label ordering is consistent across inference, prepass, and calibration. COCO JSONs are still kept (derived) for SAM3/RF‑DETR tooling.

### Glossary management
Each dataset can store a **canonical glossary** (used for SAM3 text prompts). The UI supports:
- selecting a dataset glossary,
- editing or replacing the glossary text,
- saving to dataset metadata.

You can also use a glossary library entry (named glossaries) or a custom glossary override in Deep prelabeling.

Settings per step (what the user configures)
- **Detectors (YOLO / RF‑DETR)**:
  - **Selectors**: choose one or more trained detectors to run (YOLO and/or RF‑DETR) and pick the specific trained run for each in the Deep prelabeling panel.
  - **Windowing**: SAHI window size + overlap (Deep prelabeling panel).
- **SAM3 text**:
  - **Glossary source**: dataset glossary vs glossary library vs custom text (Deep prelabeling panel).
  - **Qwen expansion**: toggle “Extend glossary with Qwen” and set max new terms per class (Deep prelabeling panel).
- **SAM3 similarity**:
  - **Exemplar min score**: choose the minimum score for exemplar selection (Deep prelabeling panel).
  - **Windowed similarity extension**: optional checkbox to add windowed similarity on top of global similarity (Deep prelabeling panel).
- **Classifier (used in later stages / calibration features)**:
  - **Selector**: choose the classifier head to generate per‑class probabilities for the calibration features (Deep prelabeling panel).
  - Note: calibration prepass keeps all candidates; the classifier is not used as a gate there.

Default thresholds (currently hard‑coded, not user‑exposed)
- Detector confidence: `0.45` — minimum confidence for YOLO/RF‑DETR detections (both full‑frame + SAHI) before they enter the candidate pool.
- SAM3 text score threshold: `0.20` — minimum SAM3 text score required for a prompt‑based detection to be kept in the prepass.
- SAM3 similarity score threshold: `0.30` — minimum SAM3 similarity score for similarity‑based detections (global and optional windowed pass).
- Similarity exemplar min score: `0.60` — minimum score for a detection to be used as an exemplar when seeding similarity search.
- SAM3 runtime score/mask: `0.20 / 0.20` — SAM3 internal score and mask thresholds used during inference; lower values favor recall.
- Dedupe IoU (both passes): `0.75` — IoU threshold used to merge overlapping detections after detectors + text, and again after similarity.
- Calibration evaluation IoU: `0.50` — IoU used when computing TP/FP/FN during calibration evaluation.
- Scoreless IoU: `0.00` — scoreless candidate filter is disabled by default (keeps all candidates).

Standardized prepass + calibration flow (current default)
1. **Detectors**: run full‑frame + SAHI for each selected detector (YOLO/RF‑DETR).
2. **SAM3 text**: run with glossary (optionally Qwen‑expanded).
3. **Dedupe pass A**: merge overlaps across detector + SAM3 text candidates.
4. **SAM3 similarity**: run global similarity (windowed extension is optional).
5. **Dedupe pass B**: merge overlaps after similarity expansion.
6. **Calibration features**: build candidate features (including classifier prob vector + per‑source scores + context counts).
7. **Calibrator**: XGBoost accept/reject (context features, log1p counts + z‑score normalization).
8. **Final output**: calibrated detections with dedupe IoU 0.75.

Calibration workflow + caching
- Calibration runs are stored under `uploads/calibration_jobs/<job_id>/`.
- Intermediate prepass + feature + label caches are stored under `uploads/calibration_cache/` and keyed by the payload hash.
- You can poll a running job via `GET /qwen/calibration/jobs/{job_id}` to track progress.

Calibration benchmark (IoU=0.50, qwen_dataset, validation split)
| Dataset size | Windowed SAM3 text | Windowed SAM3 similarity | Pipeline | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4000 | no | no | YOLO‑supported clusters (source_list contains yolo) | 0.769 | 0.532 | 0.629 |
| 4000 | no | no | RF‑DETR‑supported clusters (source_list contains rfdetr) | 0.712 | 0.562 | 0.628 |
| 4000 | no | no | YOLO + RF‑DETR (dedupe on source_list union) | 0.663 | 0.635 | 0.649 |
| 4000 | no | no | **Prepass + XGBoost (context)** | **0.850** | **0.799** | **0.824** |
| 2000 | no | no | **Prepass + XGBoost (context)** | **0.844** | **0.688** | **0.758** |

Notes:
- The detector baselines above are derived from **prepass clusters** using `source_list` membership (i.e., clusters that had detector support). This is more faithful than filtering by `score_source` alone, which only keeps clusters whose primary score came from a detector.
- “Raw detector” baselines should be measured from a detector‑only run if you want a pure comparison.

Notes:
- The IoU=0.50 evaluation is used for calibration selection (recall‑friendly for prelabeling).
- At IoU=0.75, the candidate‑pool ceiling recall from this prepass is ~0.543 (so higher recall targets require a richer prepass or looser IoU).

### Prepass smoke test (10 images)
Run a minimal smoke test locally:
```bash
bash tools/run_qwen_prepass_smoke.sh --count 10 --seed 42 --dataset qwen_dataset
```
This writes a JSONL log of detections + trace events for each image.

### Prepass benchmark (TP/FP/FN vs COCO)
Run the benchmark harness to compare detections against COCO GT:
```bash
bash tools/run_qwen_prepass_benchmark.sh --count 10 --seed 42 --dataset qwen_dataset
```
The script writes a JSONL log plus a summary JSON with per‑class TP/FP/FN.

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
- **Qwen captioning**: `Label Images` panel → `/qwen/caption` → prompt uses scene hints + optional label hints → captions can be saved into `text_labels/` in YOLO exports.
- **Deep prelabeling**: `Deep prelabeling` panel → `/qwen/prepass` → deterministic prepass + calibration → merged detections returned to UI.
- **Agent Mining (recipe search)**: Agent Mining tab → `/agent_mining/jobs` (start/poll/cancel) → results include per-class recipes in either:
  - `sam3_greedy`: prompt bank + optional crop bank (positives/negatives) + optional embedded pretrained CLIP head
  - `sam3_steps` (`schema_version=2`): explicit multi-step prompt chain (step list), plus optional embedded pretrained CLIP head
  Save/export as portable ZIP via `/agent_mining/recipes`.
- **Apply one recipe**: UI → `/agent_mining/apply_image` → `_apply_agent_recipe_to_image()`:
  - `sam3_steps`: run each step (seed → diverse seed selection → SAM3 visual expand → final CLIP filter + IoU de-dupe)
  - `sam3_greedy`: prompt bank → (optional) crop-bank CLIP filter → SAM3 expand → (optional) CLIP head gate → IoU de-dupe
- **Apply a recipe cascade**: UI → `/agent_mining/apply_image_chain` → run multiple recipes → per-class de-dupe → optional cross-class de-dupe (by group or global) with optional CLIP-head-based confidence → detections returned to UI.
- **YOLOv8 training**: Train YOLO tab → `/yolo/train/jobs` (start/poll/cancel) → runs saved under `uploads/yolo_runs/` with `best.pt` + metrics → download/delete via `/yolo/runs`.
- **RF-DETR training**: Train RF-DETR tab → `/rfdetr/train/jobs` (start/poll/cancel) → runs saved under `uploads/rfdetr_runs/` with best checkpoints + metrics → download/delete via `/rfdetr/runs`.

## Prerequisites
- Python 3.10 or newer (3.11+ recommended).
- Optional GPU with CUDA for faster CLIP/SAM inference.
- Ultralytics YOLOv8 (AGPL‑3.0) for the Train YOLO tab. Review the license terms before use: https://github.com/ultralytics/ultralytics/blob/main/LICENSE and https://www.ultralytics.com/license
- RF-DETR (Apache‑2.0) for the Train RF-DETR tab. Review the license terms before use: https://github.com/roboflow/rf-detr/blob/main/LICENSE
- Segmentation training requires polygon labels (YOLO‑SEG or COCO‑SEG). Dataset Management shows which formats are ready.
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
   Torch wheels are hardware-specific; replace `torch`/`torchvision` with the build matching your CUDA/cuDNN stack if needed. This installs Ultralytics YOLOv8 + RF-DETR for the Train YOLO/RF-DETR tabs; review the license terms linked above before using them.
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
   QWEN_MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct
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

### Dataset readiness badges (YOLO/COCO/SEG)
Dataset Management displays readiness tags so you can see when a dataset is usable:
- **YOLO** – images/labels + `labelmap.txt` detected.
- **YOLO‑SEG** – YOLO polygon labels detected (required for YOLOv8 segmentation training).
- **COCO** – COCO annotations present (used by SAM3 training + recipe mining).
- **COCO‑SEG** – COCO polygon annotations present (required for RF‑DETR segmentation training).
If a dataset only has COCO, the backend auto‑converts COCO → YOLO so the labelmap order stays consistent for inference and calibration.

### Dataset Management & Glossaries
- Upload YOLO datasets (zip) in the **Datasets** panel; the backend normalizes layout and stores metadata.
- Each dataset has a **canonical glossary** (used by SAM3 text prompts). Edit or replace it in the Dataset Manager.
- You can also maintain a **glossary library** (named glossaries) and reuse it across datasets.
- When running Deep prelabeling, choose glossary source: dataset glossary, glossary library entry, or custom text.

### Captioning with Qwen3 (Label Images tab)
1. Load an image and click **Qwen Captioning** in the Label Images panel.
2. Pick a caption model (active run or a base Qwen3 size) plus a variant (Instruct/Thinking).
3. Optionally add style prompts and opening phrases (JSON list or one per line).
4. For extra detail, use **Detailed (windowed)** — always 2×2 tiles with ~20% overlap, expanded to model input resolution.
5. Adjust **Sampling preset** (recommended defaults per variant: Instruct vs Thinking) if you want more creative diversity or deterministic outputs.
6. Click **Run caption** and optionally save the caption as a `text_labels` entry.
7. Export as YOLO ZIP: captions live under `text_labels/` next to label files.
8. **Fast mode** keeps Qwen models loaded between caption requests (faster, higher VRAM). Leave it off for max stability.

Captioning quality guardrails:
- Counts are injected as immutable facts (the model should not say “visible” vs “authoritative” counts).
- Meta language is banned (“labels,” “hints,” “bounding boxes,” etc.). If it leaks in, a cleanup pass rewrites the caption.
- Repetition/loops and truncation are detected; a minimal‑diff cleanup rewrites into a single clean sentence when needed.
- Refine mode (Thinking → Instruct) is constrained to minimal edits so it doesn’t invent new objects or actions.
- Windowed captions are merged with explicit “preserve detail” instructions (people counts, vehicles, unique objects).
- Labelmap tokens (e.g., `light_vehicle`) are explicitly discouraged in the final caption; natural language is preferred.

### Training detectors (YOLOv8 + RF-DETR)
1. Open **Train YOLO** or **Train RF-DETR**.
2. Select a dataset; the UI will only allow segmentation if polygon labels exist.
3. Choose model size and training mode (fine-tune vs from-scratch).
4. Start a run and watch the right-hand metrics card update.
5. Download `best.pt` and metrics from the Saved Runs panel.

## Updates
- 2026-01-13: Added Qwen3 captioning flow with editable style + opening lists, label-hint injection, and `text_labels/` export.
- 2026-01-13: Added YOLOv8 + RF-DETR training UIs, run tracking, and saved-run management with best checkpoints only.
- 2026-01-13: Expanded model catalog (Thinking/Instruct + FP8 options) and GPU capability warnings for large caption models.
- 2026-01-16: Captioning guardrails (fixed counts, meta‑language removal, repetition/truncation cleanup) + fast‑mode toggle for speed/VRAM tradeoff + decode presets (temp/top‑p) per variant.
- 2026-01-23: Deep prelabeling + calibration (XGBoost), glossary library, and YOLO‑first dataset auto‑conversion (COCO → YOLO).

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

### YOLOv8 Training (Detect + Segment)
1. **Prepare a dataset** in the **Dataset Management** tab (YOLO or COCO). Ensure the dataset shows the YOLO badge.
2. **Open the Train YOLO tab** and choose:
   - Task (detect vs segment),
   - Variant (YOLOv8n–x, plus P2 variants),
   - Training mode (from scratch or fine-tune),
   - Augmentations as needed.
3. **Accept Ultralytics terms** (required to start training).
   - YOLOv8 is provided by Ultralytics under AGPL‑3.0; review the license: https://github.com/ultralytics/ultralytics/blob/main/LICENSE and https://www.ultralytics.com/license
4. **Start training** and monitor progress + metrics in the right panel.
5. **Download or delete runs** from the “Saved YOLO Runs” section. We keep only `best.pt` + metrics (`results.csv`, `metrics.json`) to avoid bloating disk.

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
- The **Qwen 3 Assist** card now focuses on the object list: drop keywords such as `car, bus, kiosk` and the template pulled from the active Qwen model builds the full prompt. Choose whether the detections stay as raw boxes, go through the new “Bounding boxes → SAM cleanup” mode, or emit SAM-ready points. Expand the advanced overrides to supply a one-off custom prompt, tweak the image-type description, or add extra context for tricky scenes.
- Use **Caption current image** to generate a concise description of the scene. The captioner sends the current boxes/polygons as hints (with optional counts/coords) and automatically summarizes when there are too many detections; saved captions are written to a `text_labels/` folder in the YOLO export zip.
- Press **`X`** with a bbox selected and SAM/CLIP will refine it in place; double-tap `X` to batch-tweak the entire class (the tweak always targets whichever class is currently selected in the sidebar). *(GIF placeholder)*
- Import YOLO `.txt` folders or zipped annotation bundles via the dedicated buttons—the app now streams bboxes even while images are still ingesting.
- Auto class, SAM box/point modes, and multi-point masks share a top progress indicator and support keyboard shortcuts documented in the panel footer.

### Deep Prelabeling Tab
- Select detector runs (YOLO, RF‑DETR) and a classifier head.
- Choose glossary source and optional Qwen expansion (max new terms per class).
- Decide whether to enable windowed SAM3 similarity (global similarity always runs).
- Start a prepass + calibration job; progress is streamed and cached so repeats are fast.

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

Wide DINOv3 MLP sweep (qwen_dataset, 5 bg classes): 40 size configs across vits16/vitb16/vitl16 + SAT variants, each with/without label smoothing. Full tables live in `clip_dinov3_metrics_20241224.*`.

Top DINOv3 MLP heads (by FG macro F1):

| Encoder | MLP sizes | Label smoothing | FG macro precision | FG macro recall | FG macro F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| dinov3-vits16 | 384,192 | 0.1 | 0.8059 | 0.8551 | 0.8266 | 0.8976 |
| dinov3-vitl16 | 1024,512 | 0.1 | 0.8057 | 0.8528 | 0.8264 | 0.8977 |
| dinov3-vitl16 | 768,384 | 0.1 | 0.7991 | 0.8548 | 0.8226 | 0.8996 |
| dinov3-vitb16 | 768,384 | 0.1 | 0.7911 | 0.8521 | 0.8182 | 0.8875 |
| dinov3-vitl16 | 768 | 0.1 | 0.7873 | 0.8631 | 0.8180 | 0.8927 |

Takeaways from the sweep:
- **Soft targets (label smoothing 0.1)** improved FG macro precision by **+0.030** and recall by **+0.014** on average (macro F1 **+0.026**). Max gains reached +0.113 precision / +0.035 recall; worst-case deltas were small (≈‑0.03 precision, ≈‑0.001 recall).
- Best overall in this sweep: **dinov3-vits16 + MLP 384/192 + smoothing 0.1** (FG macro F1 **0.8266**).

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

## 2026-01-12 – Segmentation Guards + Training UX
- Enforced **segmentation-only training** for YOLOv8‑seg: bbox‑only datasets are rejected to prevent invalid runs.
- RF‑DETR segmentation now accepts YOLO‑seg datasets by auto‑converting polygons to COCO‑SEG when needed.
- Dataset Management now surfaces **YOLO‑SEG / COCO‑SEG** readiness so polygon availability is visible at a glance.
- RF‑DETR training now exposes **multi-scale + expanded scales** with clarified scale/resolution guidance.
- Added lightweight **color/blur augmentations** for RF‑DETR (HSV jitter, grayscale, blur).
- Training charts now live-update for **CLIP / classifier heads / YOLO**, plus RF‑DETR DDP log streaming.

## 2026-01-11 – RF‑DETR Integration
- Added the **RF‑DETR training tab** with job/run management and export handling.
- Added **multi‑GPU training** for RF‑DETR plus optimized export for inference.
- RF‑DETR now supports **region inference** from the Label Images panel, with a model selector.

## 2026-01-09 – YOLO Region Guidance
- Added **YOLO region inference** warnings + shortcut toasts for quicker discoverability.

## 2026-01-08 – YOLO Training Pipeline
- Added **YOLO training run management** (job list, downloads, cleanup, metrics chart).
- Added **active run selection** for YOLO inference in the Label Images panel.
- Enforced **Ultralytics TOS acceptance** and documented the dependency/licensing.

## 2025-12-25 – DINOv3 MLP Benchmarks + Soft Targets
- Benchmarked **DINOv3 + MLP** heads (small/base/large) with and without **label smoothing** and logged results to the comparison tables.
- Added a concise methodology (`classifier_testing_methodology.md`) so future classifier runs are evaluated on the same split + metrics.
- Updated the Train CLIP section with a quick summary table and the precision/recall tradeoffs from soft targets.

## 2025-12-26 – MLP Training Improvements + Benchmarks
- MLP training now supports **balanced sampling**, **mixup**, **L2-normalized embeddings**, and an optional **focal loss** path, with tuned defaults for recall-heavy behavior.
- Embedding cache signatures now prefer dataset metadata signatures (when available) to avoid cache churn caused by file mtimes.
- Refreshed the CLIP/DINOv3 benchmark sweep with mixup + balanced sampling + normalization and added a diff summary:
  - `clip_dinov3_metrics_20241224.csv`
  - `clip_dinov3_metrics_20241224.json`
  - `clip_dinov3_metrics_20241224.txt` (includes analysis)
  - `clip_mlp_diff_summary_20241224.txt`
- Added a reusable benchmark runner (`tools/run_mlp_benchmarks.py`) that can drive the backend API to reuse GPU‑cached embeddings.

## 2025-12-27 – Classifier Quality Knobs + Calibration
- Added **embedding centering/standardization**, **effective-number class weights**, **MLP activation + LayerNorm**, and **MLP hard mining** controls.
- Added optional **temperature scaling** for probability calibration (post-train, does not change argmax).
- Training jobs now log **phase timings** (scan / embed / train / calibration / save) for throughput debugging.

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

- Landed first-class Qwen 3 support: the backend now mirrors the PyImageSearch zero-shot recipe (chat templates + `process_vision_info`) and exposes `/qwen/infer` so the UI can request raw boxes, bbox→SAM cleanups, or SAM-ready click points.
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
