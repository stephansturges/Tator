/!\ Under active development. Dataset management + training + calibration workflows evolve quickly; expect periodic changes. /!\

# 🥔 Tator — Local CLIP + SAM Annotation & Prelabeling Toolkit

Tator is a single‑machine annotation stack with a browser UI and a FastAPI backend. It focuses on fast, reliable **human labeling** (CLIP + SAM assists) and a deterministic **Ensemble Detection Recipe (EDR)** workflow for high‑quality prelabeling (prepass + calibration with detectors + SAM3 + XGBoost).

We previously experimented with agentic annotation loops and removed them. Qwen is now used **only** for captioning and for optional glossary expansion used by SAM3 text prompting.

**Vision:** Tator is for teams who already have an object‑detection dataset (or need to grow one) and want to scale annotation without compromising quality. The goal is to combine strong human‑in‑the‑loop tools with deterministic automation so you can prelabel quickly, then review and correct with confidence.

---

## What you get

### Fast interactive labeling
- **Auto Class Corrector** — CLIP snaps a box to the most likely class.
- **Auto Box Refinement** — SAM tightens loose boxes; CLIP verifies the class.
- **Point‑to‑Box** — click once and SAM draws a tight box.
- **Multi‑point prompts** — add positive/negative points to sculpt tricky objects.

### Deterministic prelabeling (EDR workflow)
- **Prepass (detectors + SAM3)** builds a high‑recall candidate pool.
- **Dedupe** (IoU merge) stabilizes candidate clusters.
- **Calibration (XGBoost)** filters candidates to maximize F1 while enforcing a recall floor.
- **Glossary management** (dataset glossaries + library) and optional Qwen expansion.

### Training & model management
- **CLIP/DINOv3 head training** from the UI (cached embeddings).
- **YOLOv8 training** and **RF‑DETR training** with saved‑run management.
- **SAM3 training** from the UI (device selection + run management).
- **Qwen captioning** with windowed mode and guardrails.
- **YOLO head grafting (experimental)** to add new classes without retraining the base backbone.

---

## Repository layout
```
Tator/
├─ app/                  FastAPI app object (uvicorn imports this)
├─ api/                  Route handlers (APIRouter modules)
├─ services/             Core backend logic (prepass, calibration, detectors, sam3, qwen)
├─ utils/                Shared helpers (coords, labels, image, io, parsing)
├─ localinferenceapi.py  Router wiring + shared state (thin shim)
├─ ybat-master/          Frontend UI (static HTML/JS/CSS)
├─ tools/                CLI helpers
├─ tests/                Unit tests
└─ uploads/              Runtime artifacts + caches (git-ignored)
```

### Backend module map (where core logic lives)
- **EDR workflow (prepass + calibration)**: `services/prepass*.py`, `services/calibration*.py`
- **Detectors (YOLO / RF‑DETR)**: `services/detectors.py` + `services/detector_jobs.py`
- **Classifier heads (CLIP/DINOv3)**: `services/classifier*.py`
- **SAM3**: `services/sam3_*.py`
- **Qwen captioning + glossary expansion**: `services/qwen.py`, `services/qwen_generation.py`
- **Datasets / glossaries**: `services/datasets.py`, `utils/glossary.py`
- **Shared helpers**: `utils/coords.py`, `utils/image.py`, `utils/labels.py`, `utils/io.py`

### Backend flow (high‑level)
```
Frontend (ybat-master)
   │
   ▼
FastAPI app (app/ + localinferenceapi.py)
   │
   ▼
APIRouters (api/*) ──► Services (services/*) ──► Helpers (utils/*)
```
Migration note: `localinferenceapi.py` now acts as a thin router shim; core logic
has moved into `services/` and `utils/`.

---

## Ensemble Detection Recipe (EDR) workflow

An **Ensemble Detection Recipe (EDR)** is the full reusable detection workflow made of:

- a **prepass**, which generates a broad candidate pool
- a **calibration**, which scores and filters that candidate pool

For the fuller operator-facing explanation, see [docs/ensemble_detection_recipe_explainer.md](docs/ensemble_detection_recipe_explainer.md).

### EDR prepass architecture (detectors + SAM3 + dedupe)
```
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
- **Full‑frame + SAHI**: every detector runs twice (full‑frame + SAHI) then merges.
- **SAM3 text**: driven by the dataset glossary (optionally expanded by Qwen).
- **Similarity**: global similarity is always on; windowed similarity is optional.
- **Dedupe**: run twice (after detectors + text; after similarity).
- **Calibration**: uses `prepass_keep_all=true` (no classifier gating) so the model sees the full candidate pool.

### EDR calibration workflow + caching
- Jobs stored under `uploads/calibration_jobs/<job_id>/`.
- Intermediate prepass/features/labels cached under `uploads/calibration_cache/` keyed by payload hash.
- Poll status via `GET /calibration/jobs/{job_id}`.

### Metric taxonomy (strict tiers, IoU=0.50 default)
Calibration/evaluation now reports metrics in explicit tiers to avoid attribution ambiguity:

1. `raw_detector`: true replay baselines from raw detector atoms (YOLO-only, RF-DETR-only, YOLO+RF-DETR union).
2. `post_prepass`: candidates after detector + SAM3 text/similarity generation, before cluster dedupe.
3. `post_cluster`: candidates after dedupe/fusion.
4. `post_xgb`: final accepted candidates after XGBoost calibration thresholding.

Notes:
- Metric outputs are written under each calibration job in `uploads/calibration_jobs/<job_id>/ensemble_xgb.eval.json`.
- Prepass cache artifacts under `uploads/calibration_cache/prepass/.../images/*.json` persist atom provenance so raw baselines do not require detector replay.
- Ground truth targets are YOLO labels (for example `uploads/clip_dataset_uploads/qwen_dataset_yolo`).

### Comparison policy and acceptance gate
- Primary comparison IoU is `0.50`.
- Report two baseline groups for every run:
  - **Primary comparator (same tier):** `post_cluster.source_attributed.yolo_rfdetr_union`
  - **Diagnostics only:** `raw_detector` replay (`yolo`, `rfdetr`, `yolo_rfdetr_union`)
- Acceptance gate: the `post_xgb.accepted_all` ensemble must beat `post_cluster.source_attributed.yolo_rfdetr_union` on the agreed target metric (typically F1, with precision/recall shown).

### From fast annotation to reusable prelabeling
Tator is meant to feel like one continuous workflow: annotate faster, train from the same interface, build a stronger prelabeling recipe, and then reuse that recipe so the next round of work starts with much better suggestions.

Typical workflow in the product:

1. Use the annotation helpers to speed up manual work.
   - Auto-class correction helps snap a box to the likely label.
   - SAM refinement helps tighten loose boxes.
   - Point-to-box and multi-point prompting help with small, thin, crowded, or awkward objects.
   - The goal is simple: spend more time approving and correcting, less time drawing everything from scratch.
   - [**insert GIF: annotation helpers, box refinement, point-to-box flow**]
2. Build better dataset-specific coverage.
   - The glossary gives the system the right words for the dataset.
   - SAM3 text uses that vocabulary to find candidates the detectors may miss.
   - Similarity helps extend from strong examples to repeated look-alike objects.
   - [**insert screenshot: glossary-driven prompting and similarity-assisted suggestions**]
3. Train the supporting models directly from the browser.
   - You can launch classifier, YOLO, RF-DETR, and SAM3 training runs from the same product.
   - Runs are managed in one place instead of being split across separate annotation and training workflows.
   - [**insert screenshot: model training and run management in the UI**]
4. Build an EDR when you want stronger automation.
   - The EDR combines a broad candidate-generating prepass with a learned calibrator.
   - In plain terms: it gathers a lot of plausible detections, then learns which ones are worth keeping.
   - Once saved, that EDR becomes a reusable recipe for the dataset and configuration it was built for.
   - [**insert GIF: EDR builder / recipe creation flow**]
5. Apply the saved EDR back through the interface.
   - Future prelabeling jobs can start from the saved recipe instead of from raw detector output.
   - That makes review much faster because users are mostly cleaning up strong suggestions instead of constructing the first draft themselves.
   - [**insert screenshot: applying a saved EDR to a new job from the interface**]

### Why this is useful in practice
The value of the EDR workflow is not just that it finds more objects. It is that it makes automation easier to trust and easier to reuse.

- The prepass is broad on purpose, so it can recover objects that a plain detector-only pass would miss.
- The calibrator cuts back the false-positive tail, so that broad candidate pool becomes something a human can review efficiently.
- Windowed detection, glossary-driven SAM3 text, and similarity search all help widen coverage.
- Source-aware calibration helps keep the final output practical instead of shipping every possible candidate to the user.

In short: the system is designed to increase useful recall first, then restore precision so the result is still fast to review.

### Example result on the current experimental dataset
On the current experimental `qwen_dataset`, the canonical EDR materially improves the final accepted detections over the detector-union post-cluster baseline:

| Surface | Precision | Recall | F1 | Delta vs detector-union baseline |
|---|---:|---:|---:|---:|
| Final EDR executor | **0.9227** | **0.8062** | **0.8605** | **+0.0856** |
| Detector-union attributed post-cluster baseline | 0.6613 | 0.9356 | 0.7749 | +0.0000 |

The useful takeaway is not the exact internal recipe settings. It is that the system trades a portion of raw recall for a large precision improvement, and that trade produces a meaningfully better prelabeling result for human review.

Canonical EDR discovery now lives in:

- `tools/run_canonical_prepass_discovery.py`

This runner is the authoritative offline EDR promotion path. It reruns or reuses the sweep/ablation chain and writes the canonical reusable EDR artifacts.

Normal calibration jobs now run in **smart EDR mode** by default:

- `recipe_mode = auto`
- compute a strict fingerprint from the dataset, labelmap/glossary, detector/classifier context, windowing settings, and calibration feature version
- reuse a promoted canonical EDR for that exact fingerprint if one already exists
- otherwise build/promote a matching EDR and then train the final calibration job with it

Advanced overrides exist for:

- `reuse_only`: require an existing promoted EDR
- `force_rediscover`: rerun discovery before training

The WebUI still exposes a single main calibration button. The backend decides reuse vs discovery.

The discovery runner writes:

- `canonical_edr.json`
- `canonical_edr.md`

Legacy compatibility aliases `canonical_prepass_recipe.json` and `canonical_prepass_recipe.md` are still emitted for older code paths.

Under the chosen run root, the promoted EDR is derived from the sweep sequence rather than hand-copied defaults.


### Calibration automation helpers (current toolchain)
- `tools/build_feature_lanes_from_prepass.py`
  - Builds fixed-split calibration lanes from cached prepass artifacts.
  - Reuses precomputed labeled features when available to avoid redundant relabeling.
- `tools/run_final_calibration_sweep.py`
  - Runs coarse/refine/stack sweeps across lanes/views/seeds under a consistent evaluation policy.
- `tools/report_final_calibration_decision.py`
  - Produces a markdown decision report from ranked sweep JSON outputs.
- `tools/augment_features_with_image_context.py`
  - Injects or refreshes full-image embedding blocks into existing feature matrices for ablation work.
- `tools/tune_ensemble_thresholds_xgb.py`
  - Accepts deprecated `--relax-fp-ratio` as a compatibility no-op for older orchestration scripts.

## Feature highlights (current status)
- **Labeling assists**: auto‑class, SAM refinement, multi‑point prompts.
- **SAM variants**: SAM1/SAM2 for interactive use; SAM3 for text prompting + similarity.
- **Prepass + calibration**: deterministic prepass + XGBoost filtering (default).
- **Glossary library**: dataset glossaries + named glossary library + optional Qwen expansion.
- **Captioning**: Qwen captioning with windowed 2×2 tiles and detail‑preserving merge.
- **Training**: CLIP/DINOv3 heads, YOLOv8, RF‑DETR, SAM3.
- **Experimental**: YOLO head grafting (add new classes onto an existing YOLO run).
- **Run management**: download/delete/rename for trained classifiers and detectors.

---

## Third‑party tools & licenses (read before use)

Tator integrates several upstream tools. You are responsible for reviewing and complying with their licenses:

- **Ultralytics YOLOv8** — AGPL‑3.0
  - https://github.com/ultralytics/ultralytics/blob/main/LICENSE
  - https://www.ultralytics.com/license
- **RF‑DETR** — Apache‑2.0
  - https://github.com/roboflow/rf-detr/blob/main/LICENSE
- **Meta SAM / SAM3** — check Meta’s licenses before use
  - https://github.com/facebookresearch/segment-anything
  - https://github.com/facebookresearch/sam3 (and the Hugging Face model cards)
- **Qwen3** — review Qwen model license
  - https://huggingface.co/Qwen
- **XGBoost** — Apache‑2.0
  - https://github.com/dmlc/xgboost/blob/master/LICENSE

---

## Installation

### Prerequisites
- Python 3.10+ (3.11 recommended).
- Optional NVIDIA GPU with CUDA for faster CLIP/SAM/Qwen.
- Model weights (SAM1 required, SAM3 optional).
- If you use YOLO training/head‑grafting, you must comply with the Ultralytics license/TOS.

### Quick start
1. **Create env**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install deps**
   ```bash
   pip install -r requirements.txt
   ```
   Torch wheels are hardware‑specific; install the build matching your CUDA stack if needed.
3. **Install dev tools (optional)**
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```
4. **Model weights**
   - Place `sam_vit_h_4b8939.pth` in the repo root (SAM1).
   - Optional SAM3 setup: see below.
5. **Configure**
   ```bash
   cp .env.example .env
   ```
   Update `.env` as needed:
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
   QWEN_MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct
   QWEN_DEVICE=auto
   QWEN_MAX_NEW_TOKENS=768
   ```
6. **Run API**
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
7. **Open UI**
   - open `ybat-master/ybat.html` in a browser.

---

## Dataset management & glossaries
- Upload datasets in **Datasets** tab (YOLO preferred). COCO is auto‑converted to YOLO to preserve label order.
- Uploading the current labeling session can optionally create a **train/val split** (seeded shuffle).
- Server-path dataset workflows are explicitly split:
  - **Open transient**: temporary backend session for immediate work (expires automatically and is lost on backend restart unless saved).
  - **Save transient to library**: persists metadata + annotation overlays while keeping source files at the original path.
  - **Register path in library**: creates a persistent linked dataset record without copying source images.
- Linked datasets now expose a health status (`linked_root_status`: `ok`, `missing`, `invalid`) so moved/broken paths are visible in the UI.
- Linked dataset delete is safe: delete removes the library record + overlays only; source files are never removed.
- Each dataset has a **canonical glossary** for SAM3 text prompts.
- A **glossary library** (named glossaries) is available for reuse across datasets.
- Optional **dataset context** is stored with the dataset and used in Qwen captioning/term expansion prompts.

### Dataset and annotation API semantics
- `POST /datasets/register_path`
  - Supports strict dataset-shape validation.
  - Deduplicates by linked root/signature unless `force_new=true`.
- `POST /datasets/open_path`
  - Opens a transient session and refreshes TTL on use.
- `DELETE /datasets/transient/{session_id}`
  - Explicitly destroys transient sessions.
- Annotation write safety:
  - Snapshot/meta write endpoints require an active matching annotation lock session.
  - Sessionless/wrong-session unlock or write attempts fail with `409` (no silent unlock/write).
  - Annotation manifest responses do not expose internal server metadata paths.

---

## Captioning (Qwen)
- **Windowed mode** is always 2×2 tiles with ~20% overlap; tiles are scaled to model input size.
- The merge prompt is tuned to preserve detail across windows (people counts, vehicles, unique objects).
- Labelmap tokens are explicitly discouraged in final captions; natural language is preferred.

---

## Experimental: YOLOv8 head grafting
Head grafting lets you train a new detection head for <em>new classes</em> and merge it into an existing YOLOv8
run without retraining the backbone. This is useful when you want to expand a labelmap incrementally.

**Requirements**
- Base run must be **YOLO detect** (not segmentation) and already trained.
- New dataset must be **YOLO‑ready** and contain **only new classes**.
- Base and new labelmaps must be **disjoint** (no overlapping class names).
- Works with standard YOLOv8 and YOLOv8‑P2 variants.

**How it works**
1. Create a YOLOv8 head model for the new dataset (`nc = new classes`).
2. Load base weights, freeze backbone, and train only the new head.
3. Build a merged model with two Detect heads + a ConcatHead.
4. Append the new labelmap to the base labelmap (base classes remain intact).

**Outputs**
- `best.pt` with merged weights.
- `labelmap.txt` with base classes followed by new classes.
- Optional ONNX export.
- `head_graft_audit.jsonl` with a per‑step audit trail.
- “Head‑graft bundle” download includes `best.pt`, `labelmap.txt`, merged YAML, audit log, and `run.json`.

**Notes**
- This flow patches Ultralytics at runtime to add `ConcatHead`. Treat as experimental.
- The merged model’s class order is deterministic: base classes first, new classes last.
- Launch from the **Train YOLO** tab → **YOLO Head Grafting (experimental)** panel.
 - Inference automatically loads the runtime patch when a grafted YOLO run is activated.


## Optional: SAM3 setup
SAM3 support is optional but recommended for text prompting + similarity.

1. **Request checkpoint access** on Hugging Face.
2. **Install SAM3 repo**
   ```bash
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .
   pip install einops
   ```
3. **Authenticate**
   ```bash
   hf auth login
   ```
4. **Run API** — select SAM3 in the UI.

---

## Updates (major changes)
- **2026‑01‑23**: Removed agentic annotation loop; standardized deterministic prepass + calibration.
- **2026‑01‑27**: Calibration caching + XGBoost context features stabilized.
- **2026‑01‑29**: 4000‑image calibration evaluation (IoU=0.50) and glossary library updates.
- **2026‑01‑29**: SAM3 text prepass reworked to reuse cached image state across prompts (tile‑first).

---

## Prepass smoke test (CLI)
```bash
bash tools/run_qwen_prepass_smoke.sh --count 10 --seed 42 --dataset qwen_dataset
```

## Prepass benchmark (CLI)
```bash
bash tools/run_qwen_prepass_benchmark.sh --count 10 --seed 42 --dataset qwen_dataset
```

---

## Codebase map (where to look)
```
Tator/
├─ app/                  FastAPI app object
├─ api/                  Route handlers (APIRouter modules)
├─ services/             Core backend logic (prepass, calibration, detectors, sam3, qwen)
├─ utils/                Shared helpers (coords, labels, image, io, parsing)
├─ localinferenceapi.py  Router wiring + shared state (thin shim)
├─ ybat-master/          Frontend UI (HTML/JS/CSS)
├─ tools/                CLI helpers
├─ tests/                Unit tests
└─ uploads/              Runtime artifacts + caches
```
