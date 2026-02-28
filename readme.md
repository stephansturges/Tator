/!\ Under active development. Dataset management + training + calibration workflows evolve quickly; expect periodic changes. /!\

# 🥔 Tator — Local CLIP + SAM Annotation & Prelabeling Toolkit

Tator is a single‑machine annotation stack with a browser UI and a FastAPI backend. It focuses on fast, reliable **human labeling** (CLIP + SAM assists) and a deterministic **prepass + calibration** pipeline for high‑quality prelabeling (detectors + SAM3 + XGBoost).

We previously experimented with agentic annotation loops and removed them. Qwen is now used **only** for captioning and for optional glossary expansion used by SAM3 text prompting.

**Vision:** Tator is for teams who already have an object‑detection dataset (or need to grow one) and want to scale annotation without compromising quality. The goal is to combine strong human‑in‑the‑loop tools with deterministic automation so you can prelabel quickly, then review and correct with confidence.

---

## What you get

### Fast interactive labeling
- **Auto Class Corrector** — CLIP snaps a box to the most likely class.
- **Auto Box Refinement** — SAM tightens loose boxes; CLIP verifies the class.
- **Point‑to‑Box** — click once and SAM draws a tight box.
- **Multi‑point prompts** — add positive/negative points to sculpt tricky objects.

### Deterministic prelabeling (prepass + calibration)
- **Detectors + SAM3** build a high‑recall candidate pool.
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
- **Prepass + calibration**: `services/prepass*.py`, `services/calibration*.py`
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

## Prepass + calibration (current default)

### Prepass architecture (detectors + SAM3 + dedupe)
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

### Calibration workflow + caching
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

### Apples-to-apples benchmark snapshot (2026-02-20)
The metrics below replace earlier apples-to-oranges summaries. These runs use:

- fixed validation slice,
- IoU `0.50` for evaluation,
- identical detector targets (YOLO labels),
- strict tiered reporting (`raw_detector`, `post_prepass`, `post_cluster`, `post_xgb`).

Artifacts:

- Base evals: `tmp/emb1024_calibration_20260219_161507/nonwindow_20c8.eval.json`, `tmp/emb1024_calibration_20260219_161507/window_ceab.eval.json`
- Projection sweep: `tmp/emb1024_calibration_20260219_161507/projection_sweep/projection_sweep_report.json`
- Hybrid follow-up: `tmp/emb1024_calibration_20260219_161507/hybrid_after_sweep_jl_d512/selected_projection_hybrid_summary.json`

Primary apples-to-apples comparator (same split, IoU=0.50, same candidate tier):

| Variant | Baseline (`post_cluster.source_attributed.yolo_rfdetr_union`) | Ensemble (`post_xgb.accepted_all`) | Delta F1 |
|---|---|---|---:|
| nonwindow_20c8 | P=0.6643 R=0.9164 F1=0.7702 | P=0.9278 R=0.7048 F1=0.8011 | +0.0308 |
| window_ceab | P=0.6480 R=0.9164 F1=0.7591 | P=0.9094 R=0.6908 F1=0.7852 | +0.0260 |

Model comparison within the post-XGB tier (same split, IoU=0.50):

| Variant | Method | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| nonwindow_20c8 | XGB (1024-d embedding block) | 0.9278 | 0.7048 | **0.8011** |
| nonwindow_20c8 | XGB (JL 512 projection) | 0.9335 | 0.6984 | 0.7990 |
| nonwindow_20c8 | LR_dense + XGB_struct + blender | 0.7864 | 0.7791 | 0.7827 |
| nonwindow_20c8 | MLP_dense + XGB_struct + blender | 0.7823 | 0.7830 | 0.7827 |
| window_ceab | XGB (1024-d embedding block) | 0.9094 | 0.6908 | 0.7852 |
| window_ceab | XGB (JL 512 projection) | 0.9149 | 0.6877 | 0.7852 |
| window_ceab | LR_dense + XGB_struct + blender | 0.8824 | 0.6734 | 0.7639 |
| window_ceab | MLP_dense + XGB_struct + blender | 0.8835 | 0.6768 | 0.7665 |

Raw detector replay metrics are retained in eval artifacts as diagnostics (`metric_tiers.raw_detector.*`), but are not used as the primary acceptance comparator.

Projection sweep note:

- Best projected setup by mean F1 across both variants was `jl.d512` (mean F1 `0.7921`), but it did not beat the 1024-d XGB baseline overall.
- On the windowed variant only, `pca.d512` was marginally highest (`F1=0.7855` vs `0.7852` baseline), not a meaningful gain.

Takeaway:

- The current best overall calibrator remains single-stage XGB with the full 1024-d embedding block.
- Windowed SAM3 candidate expansion is still not translating into better post-XGB F1 under current feature/policy settings.

---

### 2000 vs 4000 extension snapshot
Using the same XGB-1024 pipeline and IoU=0.50 policy, we extended each prepass variant by +2000 images (from 2000 to 4000).

| Variant | F1@2000 | F1@4000 | Delta F1 | CovPres@2000 | CovPres@4000 | Delta CovPres |
|---|---:|---:|---:|---:|---:|---:|
| nonwindow | 0.8011 | 0.7849 | -0.0162 | 0.7503 | 0.7345 | -0.0158 |
| windowed | 0.7852 | 0.7690 | -0.0162 | 0.7267 | 0.7014 | -0.0253 |


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
- Each dataset has a **canonical glossary** for SAM3 text prompts.
- A **glossary library** (named glossaries) is available for reuse across datasets.
- Optional **dataset context** is stored with the dataset and used in Qwen captioning/term expansion prompts.

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
