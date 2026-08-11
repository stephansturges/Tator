# Tator

Tator is a local annotation-assistance workbench for object-detection datasets.
Its main job is to help you **extend an existing trusted dataset with new images
or videos while doing less manual drawing, less blind review, and less repeated
cleanup**.

The product is not a one-click label factory. Tator uses local models to rank
new media, draft boxes, suggest classes, inspect suspicious labels, and train
reusable helpers. The human still decides what enters the dataset and which
annotation changes become trusted labels.

![Dataset extension loop](docs/assets/readme-overview/01-dataset-extension-loop.png)

## Start Here

Install the local environment once:

```bash
poetry install --only-root
poetry run tator-setup macos
```
the second piece of the poetry setup contains a harcoded ref to python3.11:

python3.11 -m venv .../xxx/Tator/.venv-macos

if your Poetry setup picked a different version of python then run the venv
init with the correct version before the tator-setup step, ie:

python3.1xx -m venv .../xxx/Tator/.venv-macos


Then start the backend from the repository root with this single command:

```bash
tools/run_macos_backend.sh
```

If you are not already in the repo, use:

```bash
cd <your Tator checkout> && tools/run_macos_backend.sh
```

Leave that terminal running. When the backend is up, open:

```text
http://127.0.0.1:8000/
```

The backend serves the app at `/` and `/tator.html`. The old `/ybat.html` URL
redirects to `/tator.html`.

If port `8000` is already in use:

```bash
PORT=8080 tools/run_macos_backend.sh
```

The launcher checks the requested bind address before importing the backend. If
another process already owns the port, it prints the listener and exits instead
of repeatedly loading models and restarting into the same bind failure.

Quick health check from a second terminal:

```bash
curl http://127.0.0.1:8000/system/health_summary
```

For setup details, see [Environment Setup](docs/environment_setup.md) and
[macOS Inference Setup](docs/macos_inference_setup.md).

## What Tator Is For

Use Tator when a manual annotation loop is too slow, but blind automation is too
risky.

The strongest use case is dataset extension:

| You have... | Tator helps you... |
| --- | --- |
| A trusted accepted dataset | Keep its label map, glossary, storage state, and exports explicit. |
| New images or videos | Score the whole upload batch and keep the media most worth annotating. |
| Lots of boxes to draw | Use detectors, SAM/SAM3, class heads, and Qwen as draft assistants. |
| Existing labels that may be wrong | Find likely-wrong classes, overlap conflicts, outliers, and subclass islands. |
| A project you will repeat | Train reusable helpers and recipes from reviewed labels. |

Everything is organized around reducing human effort while preserving human
control.

## The Dataset-Extension Loop

For a normal project pass, work through the UI in this order:

| Step | Goal | Main UI area |
| --- | --- | --- |
| 1 | Define the trusted dataset, labels, glossary, storage mode, and export state. | Dataset Management |
| 2 | Decide which new media is worth labeling before drawing boxes. | Data Ingestion |
| 3 | Draft, edit, reclassify, delete, and confirm boxes in the live workspace. | Label Images |
| 4 | Audit the expanded dataset for likely mistakes and hidden subclasses. | Data Quality Explorer |
| 5 | Train helpers so the next batch starts with better proposals. | Training and recipe tabs |
| 6 | Select local models, devices, and runtime paths. | Model/runtime settings tabs |

The top navigation has many tabs because Tator covers the whole local annotation
assist loop. Think of the tabs as five core tool groups plus runtime controls,
not as unrelated features.

## Core Tool Groups

### 1. Dataset Foundation

Dataset Management is the project control point. Use it before and after the
model-assisted work so the dataset's identity and safety state are explicit.

It handles:

- opening, uploading, linking, naming, and deleting dataset records,
- label-map order and class names,
- class glossary text for ambiguous labels,
- backend-managed dataset records,
- reviewed-data exports,
- cleanup of temporary backend records.

Two storage modes matter:

| Mode | Meaning | Best for |
| --- | --- | --- |
| Linked dataset | Tator stores metadata and overlays while source files stay where they are. | Working from an existing local image folder without copying it. |
| Managed dataset | Tator owns a backend copy of images and labels. | Reopening, training on, exporting, or deleting a named backend record. |

Deleting a linked dataset record does not delete the original source images.
Deleting a managed dataset acts on the backend-managed record, so important
datasets should still be backed up outside Tator.

### 2. Data Ingestion

Data Ingestion answers the first dataset-extension question:
**which new media is worth adding to the trusted dataset?**

![Data ingestion triage](docs/assets/readme-overview/02-ingestion-triage.png)

The flow is:

1. Choose the accepted reference dataset.
2. Build or select its reference profile.
3. Upload candidate images and videos.
4. Let Tator pool the whole current upload together.
5. Rank candidates by reference novelty, within-upload coverage, and optional
   Local Vendi patch diversity.
6. Keep or discard candidates from previews.
7. Export the accepted candidate set as a ZIP or continue into annotation.

Important behavior:

- Multiple images and multiple videos are scored as one current upload batch.
- "Keep the top 20%" means the top 20% of that pooled batch, not 20% of each
  file.
- Videos are sampled into frames before scoring.
- Reference profiles can be downloaded, uploaded, and reused for later batches.
- Ingestion ranks media value. It does not certify annotation correctness.

### 3. Assisted Annotation

Label Images is the live annotation workspace. This is where boxes are drawn,
edited, reclassified, deleted, and saved.

![Assisted annotation](docs/assets/readme-overview/03-assisted-annotation.png)

Tator can help with:

- detector proposals for first-pass boxes,
- SAM/SAM3 prompts for interactive object help,
- class predictors for class suggestions,
- Qwen captions and visual context,
- configurable keyboard shortcuts for normal keyboards and programmable
  keypads,
- reviewed-label exports.

The rule is intentionally simple: **models propose, the user reviews, and only
reviewed labels become trusted labels**.

`Shift+Y` exports the active YOLO labels and text captions. For backend-managed,
linked, and transient datasets, Tator captures an immutable annotation revision
in a background job and hands the completed labels-only archive to the browser
download manager. Repeating an unchanged export reuses its ready snapshot, while
retention limits remove older snapshots automatically. The archive includes all
stored caption alternatives, but not Data Quality Explorer vignettes, model
traces, images, or analysis artifacts.

### 4. Quality Audit And Repair

Data Quality Explorer audits labels that already exist. It embeds object crops,
projects them into 2D plots, and finds objects that are outliers, overlap
suspiciously, or appear to belong to a hidden subclass.

Its built-in recipes are fixed and comparable: Thorough combines DINOv3
tight/context features, SAM3 mask features, bounded SALAD token aggregation,
and the additional quality-ranking passes; Precise keeps the compact fused
descriptor without the heaviest ranking branches; Fast uses a single DINOv3
view for quick map iteration. Custom exposes the same dimensions, weights,
crop, pooling, and projection controls explicitly. PCA and UMAP only arrange
the graph; review evidence remains in the full embedding space.

Optional 2-stage refinement keeps the pooled Stage-1 discovery result and raw
candidate queue intact, then spatially qualifies only rough candidates with
patch-level class and overlap evidence. It may change the default vignette
queue, but it never changes annotations and remains advisory to the local VLM
and human reviewer.

![Quality audit and Qwen review](docs/assets/readme-overview/04-quality-audit-qwen.png)

Use it to:

- inspect all-class structure,
- inspect one class for possible subclasses,
- switch between projection modes for different review goals,
- review likely-wrong vignettes,
- confirm the current class, skip a case, reassign a class, or jump back to the
  source image,
- use the same confirm, skip, class-change, bbox-deletion, and VLM-review
  controls from either a queue vignette or the selected object beside the graph,
- ask Qwen to review a suspicious object with crop, source-context, overlap,
  similar-example, glossary, scale, embedding, and cue evidence.

For Qwen review, the local VLM's final judgment is the core product behavior.
Deterministic checks such as overlap, edge clipping, scale, embedding distance,
and cue verification are guardrails and audit evidence. They may block automatic
mutation, but they do not replace visual reasoning.

`Class Split` remains in some internal route names for compatibility; the
current UI name is Data Quality Explorer.

### 5. Reusable Helpers And Training

Training is optional during early manual review. It becomes useful once a
project has enough reviewed labels to teach repeatable helpers.

![Reusable helpers](docs/assets/readme-overview/05-reusable-helpers.png)

Common helper paths:

| UI area | What it helps with |
| --- | --- |
| Train Class Predictor | Faster class suggestions from project-specific class heads. |
| Train YOLO / Train RF-DETR | Detector proposals and future prelabeling. |
| Train SAM3 | Promptable segmentation helpers and SAM3 datasets. |
| Train Qwen 3 | Local VLM model management and adapter-training paths. |
| Detection Recipes / SAM3 Recipe Mining | Repeatable prelabeling recipes. |
| SAM3 Vocabulary Explorer | Prompt vocabulary and class-language inspection. |

Train the helper that removes the next real bottleneck. You do not need to use
every training tab on every project.

## Runtime And Model Controls

Tator supports multiple local runtimes because macOS inference, Linux training,
and CUDA training have different dependency constraints.

Recommended setup commands:

```bash
# Apple Silicon inference and local MLX paths
poetry run tator-setup macos

# General Linux backend and training stack
poetry run tator-setup linux

# Pinned Falcon CUDA 11.8 stack
poetry run tator-setup falcon-cu118
```

Useful setup options:

```bash
poetry run tator-setup macos --dry-run
poetry run tator-setup linux --dev
poetry run tator-setup falcon-cu118 --venv-dir .venv-falcon
poetry run tator-setup macos --recreate
```

Optional macOS overrides can go in `.env.macos`:

```bash
QWEN_DEVICE=auto
QWEN_INFERENCE_PLATFORM=auto
# Optional override; by default Apple Silicon general Qwen inference uses AEON Qwen3.6 27B FP4.
QWEN_MLX_MODEL_NAME=AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4
# auto lazy-loads large MLX VLMs such as AEON; set false to force eager load.
QWEN_MLX_LAZY_LOAD=auto
# Captioning may run many model calls; implicit caption runs keep the compact MLX default.
QWEN_MLX_CAPTION_MODEL_NAME=mlx-community/Qwen3-VL-4B-Instruct-4bit
# AEON and other larger MLX models remain selectable in the UI for explicit captioning runs.
# Optional CUDA/Transformers override; default is the matching AEON NVFP4 sibling.
QWEN_MODEL_NAME=AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
# Training remains separate; default stays on a trainable Qwen3-VL checkpoint.
QWEN_TRAINING_DEFAULT_MODEL=Qwen/Qwen3-VL-4B-Instruct
TATOR_QWEN_PROGRESS_STALE_SECONDS=1800
DINOV3_BACKEND=auto
```

See [macOS Inference Setup](docs/macos_inference_setup.md) for MLX-DINOv3,
MLX-SAM, Qwen MLX-VLM, and Apple Silicon fallback behavior.
`TATOR_QWEN_PROGRESS_STALE_SECONDS` controls how long an active Qwen/prepass
progress record may sit without a heartbeat before the backend releases the UI
state and reports it as stale. The default is 30 minutes.

## Data Safety Model

Tator is designed around local, human-controlled dataset work:

- The currently open annotation workspace is the live review state.
- Label changes are advisory until the user accepts or applies them.
- Data Ingestion workspace uploads require names so temporary backend records
  can be recognized and cleaned later.
- Linked dataset deletion does not delete original source images.
- Managed dataset deletion acts on the backend-managed record.
- Long-running uploads and jobs use observable metadata rather than disappearing
  silently.
- Data Quality Explorer and Qwen review artifacts preserve raw model inputs,
  outputs, and guardrail evidence for auditability.

## Documentation Map

Use these docs when the README is not enough:

- [Environment Setup](docs/environment_setup.md)
- [macOS Inference Setup](docs/macos_inference_setup.md)
- [Ensemble Detection Recipe Explainer](docs/ensemble_detection_recipe_explainer.md)
- [Qwen Caption Run Policies](docs/qwen_caption_run_policies.md)
- [Tools Command Index](tools/README.md)

## Licenses And Model Terms

This repo is local tooling. Check the license and acceptable-use terms for every
model, dataset, and generated artifact you use.

Notable external dependencies and model families include:

- Meta SAM / SAM3 checkpoints and dependencies
- Qwen/Qwen3-VL and compatible local VLM checkpoints
- Ultralytics YOLO
- RF-DETR
- CLIP, DINOv3, C-RADIO, and related embedding backbones
- MLX, MLX-VLM, and optional Apple Silicon model ports

License compliance for trained models and exported datasets remains the user's
responsibility.
