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

---

## Repository layout
```
Tator/
├─ app/                  FastAPI app object (uvicorn imports this)
├─ localinferenceapi.py  Backend endpoints + orchestration
├─ ybat-master/          Frontend UI (static HTML/JS/CSS)
├─ tools/                CLI helpers
├─ tests/                Unit tests
└─ uploads/              Runtime artifacts + caches (git-ignored)
```

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
- Poll status via `GET /qwen/calibration/jobs/{job_id}`.

### Calibration benchmark (IoU=0.50, qwen_dataset, validation split)
| Dataset size | Windowed SAM3 text | Windowed SAM3 similarity | Pipeline | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4000 | no | no | YOLO‑supported clusters (source_list contains yolo) | 0.769 | 0.532 | 0.629 |
| 4000 | no | no | RF‑DETR‑supported clusters (source_list contains rfdetr) | 0.712 | 0.562 | 0.628 |
| 4000 | no | no | YOLO + RF‑DETR (dedupe on source_list union) | 0.663 | 0.635 | 0.649 |
| 4000 | no | no | **Prepass + XGBoost (context)** | **0.850** | **0.799** | **0.824** |
| 2000 | no | no | **Prepass + XGBoost (context)** | **0.844** | **0.688** | **0.758** |

Notes:
- Detector baselines above are derived from **prepass clusters** using `source_list` membership (more faithful than `score_source` alone).
- IoU=0.50 is used for calibration selection (recall‑friendly for prelabeling). For stricter downstream evaluation, use IoU=0.75 or higher.
- **Windowed cost**: enabling 2×2 windowed SAM3 text+similarity increased throughput from ~20 s/img to ~31 s/img (≈+55%) on dual A6000s in our qwen_dataset runs. Expect variation by GPU/model.

---

## Feature highlights (current status)
- **Labeling assists**: auto‑class, SAM refinement, multi‑point prompts.
- **SAM variants**: SAM1/SAM2 for interactive use; SAM3 for text prompting + similarity.
- **Prepass + calibration**: deterministic prepass + XGBoost filtering (default).
- **Glossary library**: dataset glossaries + named glossary library + optional Qwen expansion.
- **Captioning**: Qwen captioning with windowed 2×2 tiles and detail‑preserving merge.
- **Training**: CLIP/DINOv3 heads, YOLOv8, RF‑DETR, SAM3.
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
- Each dataset has a **canonical glossary** for SAM3 text prompts.
- A **glossary library** (named glossaries) is available for reuse across datasets.

---

## Captioning (Qwen)
- **Windowed mode** is always 2×2 tiles with ~20% overlap; tiles are scaled to model input size.
- The merge prompt is tuned to preserve detail across windows (people counts, vehicles, unique objects).
- Labelmap tokens are explicitly discouraged in final captions; natural language is preferred.

---

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
├─ localinferenceapi.py  Backend endpoints + orchestration
├─ ybat-master/          Frontend UI (HTML/JS/CSS)
├─ tools/                CLI helpers
├─ tests/                Unit tests
└─ uploads/              Runtime artifacts + caches
```
