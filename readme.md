# Tator

Tator is a local annotation-assistance stack for building and extending object
detection datasets. It combines a static browser UI with a FastAPI backend so
you can label from scratch, train dataset-specific helpers, prelabel new images,
and keep the final review loop human-controlled.

The core idea is simple: drawing every box by hand is the slow path. Tator keeps
manual labeling available, but adds CLIP/DINO class prediction, SAM/SAM3 prompt
tools, detector passes, Qwen captions, dataset glossaries, recipe mining, and
calibrated prelabeling so extending a dataset can become mostly review and
correction work instead of first-draft annotation.

## Quick Start

```bash
poetry install --only-root
poetry run tator-setup macos   # or: poetry run tator-setup linux
cp .env.macos.example .env.macos
tools/run_macos_backend.sh
```

Open `ybat-master/ybat.html` in a browser. The UI defaults to
`http://localhost:8000`.

<details>
<summary>Workflow demos to add</summary>

Tator lets you use SAM to refine bounding boxes. Draw a rough box, run SAM, and
turn the rough annotation into a tighter object outline before accepting it.

[Place GIF of SAM bounding-box refinement here]

Tator lets you turn one labeled dataset into the context for the next labeling
session. Register or upload a dataset, inspect its health, edit its label
glossary, and open it for annotation without losing track of where the images
came from.

[Place GIF of dataset registration and glossary editing here]

Tator lets you use Qwen as visual context while you label. Generate captions or
windowed captions, feed that context back into the annotation flow, and keep the
final boxes under human control.

[Place GIF of Qwen captioning inside the labeling flow here]

Tator lets you use SAM3 with project vocabulary instead of generic prompts.
Explore glossary terms, expand prompt language, and use text or visual prompts
to find objects that ordinary detector passes miss.

[Place GIF of SAM3 vocabulary or prompt exploration here]

Tator lets you package repeatable prelabeling logic as an Ensemble Detection
Recipe. Build broad candidate pools, calibrate them against reviewed labels, and
reuse the recipe on the next image batch.

[Place GIF of EDR building, calibration, or application here]

Tator lets you train and switch between local helpers from the UI. CLIP/DINO,
SAM3, YOLO, RF-DETR, and Qwen workflows can all be selected close to the review
loop instead of living in separate scripts.

[Place GIF of model training and active-model selection here]

</details>

## What Tator Helps With

- Start a dataset from raw images with a browser labeling UI.
- Extend an existing YOLO/YOLO-seg dataset without redrawing everything.
- Keep source datasets in place through linked-dataset records when copying data is undesirable.
- Use a dataset glossary so text-prompted systems speak in the same labels and synonyms as the project.
- Generate first-pass labels with detectors, SAM3 text/similarity, Qwen context, and optional Falcon proposals.
- Train and select local helper models from the same interface.
- Audit class purity with CLIP/DINOv3 embedding maps so outliers and likely
  wrong-class objects can be inspected and corrected.
- Package reusable prelabeling recipes so the next batch starts from stronger suggestions.

Tator is not meant to remove review. It is meant to automate the repetitive
80 percent: candidate generation, class suggestions, box tightening, repeated
object discovery, and first-pass dataset extension. The human stays responsible
for the final labels.

<details>
<summary>Workflow/API map</summary>

| Workflow | UI area | Backend/API surface | Primary code |
|---|---|---|---|
| Manual and assisted labeling | Label Images | `/predict_base64`, `/sam_*`, `/sam3/*_prompt`, `/yolo/predict_*`, `/rfdetr/predict_*` | `ybat-master/`, `localinferenceapi.py`, `api/sam3_prompts.py`, `services/detectors.py` |
| Dataset library | Dataset Management | `/datasets/*`, `/glossaries/*`, `/qwen/dataset/*`, `/sam3/datasets/*`, `/segmentation/build/*` | `api/datasets.py`, `services/datasets.py`, `services/segmentation.py`, `utils/coco.py`, `utils/glossary.py` |
| Qwen captions and VLM inference | Qwen Models, Label Images | `/qwen/status`, `/qwen/settings`, `/qwen/caption`, `/qwen/infer`, `/qwen/prepass` | `services/qwen*.py`, `qwen_agent_llm.py`, `qwen_agent_tools.py` |
| Class predictor training | Train Class Predictor | `/clip/train`, `/clip/classifiers/*`, `/clip/active_model` | `services/classifier*.py`, `services/clip_runtime.py`, `services/dinov3_runtime.py` |
| Class split and outlier audit | Class Split Explorer | `/class_analysis/*`, `/clip/backbones` | `api/class_analysis.py`, `localinferenceapi.py`, `ybat-master/plotly-2.35.2.min.js`, `ybat-master/ybat.js` |
| Detector training and inference | Train YOLO, Train RF-DETR, Detector Selection | `/yolo/*`, `/rfdetr/*`, `/detectors/default` | `services/detectors.py`, `services/detector_jobs.py` |
| SAM3 model workflows | Train SAM3, SAM Model Selection | `/sam3/models/*`, `/sam3/train/*`, `/sam3/storage/*` | `services/sam3_*.py`, `sam3_local/` |
| SAM3 vocabulary and recipes | SAM3 Vocabulary Explorer, SAM3 Recipe Mining | `/sam3/prompt_helper/*`, `/agent_mining/*`, `/prepass/recipes/*` | `services/prompt_helper*.py`, `services/agent_cascades.py`, `services/prepass_recipes.py` |
| EDR prelabeling and calibration | EDR Builder, Label Images | `/qwen/prepass`, `/calibration/jobs/*`, `/edr/packages/*`, `/auto_label/jobs/*` | `services/prepass*.py`, `services/calibration*.py`, `services/edr_packages.py`, `services/auto_labeling.py` |
| Runtime and system controls | Backend Config, SAM Predictors | `/system/*`, `/runtime/unload`, `/predictor_settings`, `/sam_slots`, `/sam_preload` | `api/system.py`, `services/runtime_unload.py`, `localinferenceapi.py` |

</details>

## Repository Map

```text
Tator/
  app/                    FastAPI app export for uvicorn
  api/                    APIRouter builders for each endpoint family
  services/               Backend business logic and model runtimes
  utils/                  Shared image, label, parsing, GPU, COCO, and IO helpers
  models/                 Pydantic schemas and recipe model helpers
  ybat-master/            Static browser UI
  tools/                  Setup, validation, calibration, EDR, benchmark, and debug CLIs
  tests/                  Unit, integration, and UI contract tests
  docs/                   Focused reports and setup notes
  sam3_local/             Local SAM3 training config/extensions
  constraints/            Locked optional environment constraints
  uploads/                Runtime datasets, jobs, caches, and model artifacts; git-ignored
```

`localinferenceapi.py` is still the central wiring module. New endpoint handlers
are split into `api/`, while most reusable behavior lives in `services/` and
`utils/`.

## Dataset Management Philosophy

Datasets are the center of Tator. The system works best when every automation
step is tied back to a concrete dataset record, labelmap, glossary, and optional
context note.

Tator supports three dataset patterns:

- **Upload a dataset** when you want Tator to own a copy. YOLO and YOLO-seg are
  the preferred internal formats; COCO uploads are converted where possible.
- **Open a server path transiently** when you want to inspect or label a local
  folder without committing it to the library.
- **Register a linked dataset** when source images should stay where they are.
  Tator stores metadata, overlays, glossaries, and labels, but linked dataset
  deletion does not remove source files.

Dataset records also carry:

- labelmap and class order
- linked-root health (`ok`, `missing`, or `invalid`)
- canonical glossary text for SAM3 and Qwen-assisted prompting
- optional dataset context for captions, prompt expansion, and recipe building
- annotation lock state so concurrent writes fail loudly instead of silently
  clobbering labels

This is why Tator is useful for extension work. Once a dataset has a glossary,
active helpers, and one or more saved recipes, new image batches can start from
dataset-specific suggestions rather than generic detector output.

## Assisted Labeling

The Label Images tab is the everyday workspace.

- Draw boxes manually and export labels.
- Ask CLIP/DINO heads to suggest or correct classes.
- Use SAM point, box, and multi-point prompts to tighten boxes or derive masks.
- Use SAM3 visual and text prompts when the SAM3 runtime is available.
- Run YOLO or RF-DETR full-frame, region, or windowed inference.
- Apply saved SAM3 recipes, cascades, prepass recipes, or EDR packages to the
  active image.
- Use Qwen captions as visual context while keeping final labels editable.
- Use Class Split Explorer to embed one class or all classes in the current
  annotation dataset, inspect cluster structure, and jump from suspicious
  points back to the source bbox for correction.
- Export selected crops through the chunked crop ZIP endpoints.

The output remains an annotation draft. The UI is built around fast accept,
correct, and reject loops.

The top-right `Dark` button toggles the standard dark theme. Double-click the
same button to switch into the hidden Pip-Boy-inspired terminal skin. In Pip-Boy
mode, single-click the `Pip` button to switch between green and amber; double-click
it to return to the previous light/dark theme.

## Qwen

Qwen is the local VLM path. In this repo it is used for:

- image and windowed captions
- structured `/qwen/infer` calls
- dataset-aware context for prepass runs
- optional glossary and prompt expansion support
- Qwen dataset upload/build workflows
- Qwen model activation, settings, unload, and training job endpoints
- optional qwen-agent adapters for local tool-calling experiments

The portable default model name is `Qwen/Qwen3-VL-4B-Instruct`. On Apple
Silicon, `QWEN_INFERENCE_PLATFORM=auto` selects MLX-VLM when the `mlx-vlm`
package is installed and no adapter checkpoint is active. That path maps Qwen3
VL model choices to quantized `mlx-community` models, with
`mlx-community/Qwen3-VL-4B-Instruct-4bit` as the default. Use
`QWEN_INFERENCE_PLATFORM=transformers` to force the PyTorch/Transformers path,
or `QWEN_INFERENCE_PLATFORM=mlx_vlm` to require MLX-VLM.

The UI exposes the same runtime controls under **Backend Config -> Qwen Runtime
(advanced)**. The caption, prepass, and EDR controls can also select exact
quantized MLX model IDs directly; when an exact ID is selected, the request is
sent as that exact model rather than being rewritten by the Instruct/Thinking
variant dropdown.

The Qwen Models tab also includes CUDA/Transformers presets for official
Qwen3-VL checkpoints, curated FP8/AWQ/GPTQ checkpoints, and current compatible
abliterated Qwen3-VL checkpoints. Quantized presets are inference checkpoints;
dense and MoE entries can be selected for LoRA/QLoRA training, and quantized
CUDA selections resolve to the matching full Transformers checkpoint before
training starts. QLoRA then applies bitsandbytes NF4 4-bit quantization at
training load time. MLX model presets are also trainable on Apple Silicon
through the MLX-VLM trainer; quantized MLX checkpoints use the same LoRA adapter
path as QLoRA-style training.

Model access, memory requirements, and device support still depend on your
machine and Hugging Face cache/authentication. Large Thinking and MoE models can
be listed in the UI even when the local hardware cannot comfortably run them.

<details>
<summary>Detection-informed captioning</summary>

Detection-informed captioning is Tator's captioning mode for dataset extension
work. A normal VLM caption can be fluent while still missing the objects that
matter to a specific dataset. Tator instead uses detector and annotation output
as priors, then asks Qwen to describe the image while keeping those salient
objects in view.

The detections are not pasted in as final prose. They are converted into a
structured captioning context:

- class names are normalized through the active dataset glossary so project
  labels become broad natural terms, such as "small vehicle" or "storage tank";
- class counts can be supplied as authoritative inventory when the user wants
  the caption to retain every detected class, and count-bearing captions are
  checked for exact numeric counts rather than just generic class mentions;
- box coordinates can be supplied when spatial grounding is useful, but the
  model is told not to mention coordinates or that hints were provided;
- `max boxes` limits how many boxes are sent when the detection set is large;
- overlapping window crops receive clipped local boxes, while backend source
  identity is used only for deduplication guidance and is never exposed as bbox
  IDs in prompts or final captions;
- the labelmap glossary explains what a class can mean, without forcing the
  model to pick an unsupported subtype from the glossary.

The captioning workflow has several variants:

- **Full-image captioning** sends the whole image plus optional detection
  priors to Qwen and asks for a final caption in one pass.
- **Windowed captioning** first looks at crop windows, then merges those local
  observations into a full-image caption. When label hints are present, the
  default is to focus on windows that intersect labeled objects; empty windows
  are only used as broad scene context when explicitly enabled.
- **Spatial grounding for windows** carries each crop's approximate full-image
  region and percent bounds into later full-image and merge prompts. Local crop
  phrases such as "bottom-right" are treated as crop-relative and must be
  rewritten in full-image terms before they reach the final caption.
- **Two-stage refine** lets Thinking models produce a draft and then uses an
  editor pass to turn that draft into a cleaner final caption. It runs once on
  full-image composition after any window observations have been collected; it
  does not run once per crop, and it is separate from the window merge pass.
- **Refinement model selection** can use the same model as captioning or a
  separate model, which is useful when a Thinking model sees details well but
  an Instruct model is better at producing clean final prose.
- **Final length controls** let the operator choose whether the final caption
  should be short or allowed to preserve many sentences of detail for large,
  dense images.
- **Prompt stack editing** exposes the caption prompt families in closed UI
  pop-downs: user request, system prompt, detection context, window prompts,
  draft/refine, window merge, and cleanup/guard instructions. These fields are
  editable and are sent with caption requests; runtime image values such as boxes,
  counts, crop windows, and model outputs are filled at generation time and are
  summarized in the Caption workflow grid.

After generation, Tator checks and repairs the caption before returning it:

- repeated loops, meta text, reasoning leakage, and incomplete endings are
  removed or sent through cleanup passes;
- missing detected classes, missing numeric counts, and count conflicts can
  trigger a coverage refine;
- non-English rewrite is reserved for real non-Latin script output rather than
  harmless punctuation differences;
- window observations are treated as supporting evidence, not a separate object
  inventory;
- a final glossary stabilizer demotes unstable subtypes back to broad class
  terms when windows, drafts, or cleanup passes disagree.

The goal is a caption that remains grounded in the image while retaining the
objects the dataset actually cares about. Detection gives Qwen a map of what is
salient; Qwen then describes those objects in natural language, adds visual
context where it can, and avoids drifting into generic captions that are fluent
but unhelpful for the dataset.

</details>

## Training and Model Management

Tator keeps helper models close to the annotation workflow:

- **CLIP/DINOv3 heads**: fast class predictors trained from managed datasets.
- **YOLOv8**: detector training with CUDA, Apple MPS, or CPU acceleration,
  active run selection, run summaries, downloads, deletion, and experimental
  head grafting for disjoint new classes.
- **RF-DETR**: detector training, active run selection, summaries, downloads,
  deletion, and full/region/windowed inference.
- **SAM3**: dataset conversion, training jobs, model registry, active model
  selection, and run promotion/deletion.
- **Qwen**: model registry, settings, cache management, and training jobs.

Heavy jobs acquire an automation lock in the UI so competing GPU-heavy actions
do not run over each other.

## EDR: Short Version

An Ensemble Detection Recipe (EDR) is a reusable prelabeling pipeline for a
specific dataset fingerprint. It generates a broad candidate pool, scores and
filters that pool, and packages the result so a future batch can reuse the same
proven setup.

Use EDR when a project is past the first few labels and you want repeatable
automation for extension batches.

<details>
<summary>EDR details</summary>

### What an EDR Contains

An EDR combines:

- a prepass recipe
- detector and SAM3 source settings
- candidate dedupe/fusion policy
- calibration feature settings
- XGBoost/MLP calibration outputs
- threshold and policy choices
- expected metrics and comparison metadata
- enough package metadata to replay the runtime later

### Prepass Structure

```text
Image
  -> detector seeds
     -> YOLO full-frame and optional SAHI/windowed passes
     -> RF-DETR full-frame and optional SAHI/windowed passes
  -> SAM3 text prompts from glossary terms and optional expansions
  -> dedupe/fusion with provenance
  -> SAM3 similarity expansion from confident exemplars
  -> final candidate pool
```

The prepass is intentionally broad. It should recover objects that a single
detector pass misses, especially small objects, boundary cases, unusual classes,
and repeated look-alike objects.

### Calibration

Calibration turns the broad candidate pool into a practical review set. It
builds feature rows from source provenance, detector/SAM scores, agreement,
geometry, cluster structure, and optional classifier signals. The current
default path uses XGBoost plus threshold tuning.

The calibrator does not create new objects. It decides which generated
candidates are worth sending to the reviewer.

### Packaging and Reuse

EDR packages are handled through `/edr/packages/*`. Saved packages can be
exported, imported, and applied to later jobs. Canonical discovery and repair
tools live in `tools/run_canonical_prepass_discovery.py` and
`tools/backfill_edr_packages.py`.

For a longer operator explanation, see
[docs/ensemble_detection_recipe_explainer.md](docs/ensemble_detection_recipe_explainer.md).

</details>

## Environment Setup

Use Poetry as the setup front door, then choose the profile that matches the
machine. Poetry only installs the lightweight `tator-setup` command; the actual
runtime dependencies still come from the profile-specific requirement files so
macOS, Linux, and Falcon CUDA do not fight over one incompatible lockfile.

```bash
poetry install --only-root

# Apple Silicon inference + Qwen MLX adapter training
poetry run tator-setup macos

# General Linux backend/training stack
poetry run tator-setup linux

# Pinned Falcon CUDA 11.8 stack
poetry run tator-setup falcon-cu118
```

Useful checks and variants:

```bash
poetry run tator-setup macos --dry-run
poetry run tator-setup linux --dev
poetry run tator-setup falcon-cu118 --venv-dir .venv-falcon
```

The direct shell setup scripts remain available and delegate to the same setup
implementation:

```bash
tools/setup_venv_macos_inference.sh
bash tools/setup_venv_falcon_cu118.sh
```

## Linux Setup

Use Linux for full training workflows, CUDA acceleration, Falcon automatic
labeling, and the broadest package compatibility.

### General Linux Backend

```bash
poetry install --only-root
poetry run tator-setup linux
source .venv/bin/activate
cp .env.example .env
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `ybat-master/ybat.html` in a browser. The UI defaults to
`http://localhost:8000`.

### Linux Model Assets

Set paths and model names in `.env`:

```bash
LOGREG_PATH=./my_logreg_model.pkl
LABELMAP_PATH=./my_label_list.pkl
CLIP_MODEL_NAME=ViT-B/32
SAM_MODEL_TYPE=vit_h
SAM_CHECKPOINT_PATH=./sam_vit_h_4b8939.pth
SAM_VARIANT=sam1
SAM3_MODEL_ID=facebook/sam3
SAM3_PROCESSOR_ID=facebook/sam3
SAM3_CHECKPOINT_PATH=
SAM3_DEVICE=auto
QWEN_MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct
QWEN_DEVICE=auto
QWEN_INFERENCE_PLATFORM=auto
QWEN_MLX_MODEL_NAME=mlx-community/Qwen3-VL-4B-Instruct-4bit
QWEN_MLX_DEFAULT_QUANTIZATION=4bit
QWEN_MAX_NEW_TOKENS=768
```

SAM1 interactive prompts require a local SAM checkpoint. SAM3 and Qwen may
download model assets from Hugging Face on first use, depending on your cache
and authentication.

### Optional SAM3 on Linux

`requirements.txt` keeps SAM3 optional because access and runtime constraints
vary by machine. Install it when you want SAM3 text/visual prompting,
similarity, training, or recipe mining:

```bash
python -m pip install "git+https://github.com/facebookresearch/sam3.git" "einops>=0.7,<1.0"
hf auth login
```

### Optional Falcon GPU Stack

Falcon automatic labeling is the most version-sensitive path. Use the pinned
CUDA 11.8 wheel setup before changing system CUDA or drivers:

```bash
poetry run tator-setup falcon-cu118
```

See [docs/environment_setup.md](docs/environment_setup.md) for the exact stack
and driver notes.

## macOS Apple Silicon Setup

The macOS path targets interactive annotation assistance with PyTorch MPS for
CLIP/SAM/SAM3/detectors and MLX-VLM for Qwen when available. YOLOv8 training
can also use PyTorch MPS on Apple Silicon through the Train YOLO accelerator
selector. RF-DETR and full SAM training remain Linux/CUDA-first, while Qwen MLX
LoRA adapter jobs are available for small enough local Apple Silicon models.

```bash
poetry install --only-root
poetry run tator-setup macos
cp .env.macos.example .env.macos
tools/run_macos_backend.sh
```

The macOS venv installs CLIP, SAM1, SAM3, YOLO, RF-DETR, Qwen Transformers,
MLX, MLX-VLM, and the local MPS compatibility path. The setup script installs
MLX-VLM with `--no-deps` after the main requirements so Qwen3 MLX support does
not force a Transformers 5 or OpenCV 4.12 stack over the rest of the backend.
The runner defaults to:

```bash
TATOR_INFERENCE_DEVICE=auto
TATOR_ALLOW_MPS=1
PYTORCH_ENABLE_MPS_FALLBACK=1
SAM3_DEVICE=auto
YOLO_INFER_DEVICE=auto
RFDETR_INFER_DEVICE=auto
QWEN_DEVICE=auto
QWEN_INFERENCE_PLATFORM=auto
QWEN_MLX_MODEL_NAME=mlx-community/Qwen3-VL-4B-Instruct-4bit
```

See [docs/macos_inference_setup.md](docs/macos_inference_setup.md) for MPS
runtime details and known fallback behavior.

## API Families

The backend exposes these endpoint groups:

- `/datasets/*`, `/glossaries/*`: dataset library, linked paths, transient
  sessions, annotation locks, text labels, glossaries, downloads.
- `/segmentation/build/jobs/*`: bbox-to-segmentation dataset build jobs.
- `/sam_*`, `/sam3/*_prompt`, `/sam3/models/*`, `/sam3/train/*`,
  `/sam3/prompt_helper/*`: SAM and SAM3 prompt/model/training/vocabulary flows.
- `/qwen/*`: Qwen status, settings, model activation, captions, inference,
  prepass, dataset upload, and training.
- `/clip/*`: class predictor training, active model, classifier registry,
  labelmap registry, downloads, rename/delete.
- `/fs/upload_classifier`, `/fs/upload_labelmap`, `/crop_zip_*`,
  `/predict_base64`: support endpoints for uploaded model assets, crop exports,
  and basic CLIP class prediction.
- `/yolo/*`, `/rfdetr/*`, `/detectors/default`: detector inference, training,
  active run selection, summaries, downloads, deletion, and YOLO head grafting.
- `/prepass/recipes/*`, `/calibration/jobs/*`, `/edr/packages/*`,
  `/auto_label/jobs/*`, `/agent_mining/*`: recipe mining, calibrated prelabeling,
  EDR packaging, and automatic-labeling jobs.
- `/system/*`, `/runtime/unload`, `/predictor_settings`, `/sam_slots`,
  `/sam_preload`: health, GPU/storage status, runtime unload, and SAM predictor
  slot management.

Run this to inspect the live OpenAPI surface:

```bash
curl http://127.0.0.1:8000/openapi.json
```

## Tools

Common commands:

```bash
BASE_URL=http://127.0.0.1:8000 SKIP_GPU=1 tools/run_refactor_validation.sh
BASE_URL=http://127.0.0.1:8000 SKIP_GPU=1 tools/run_fuzz_fast.sh
python tools/scan_unused_defs.py
python tools/run_openapi_missing_param_sanity.py
python tools/run_openapi_missing_query_sanity.py
```

Tool groups:

- setup: `tools/setup_venv_falcon_cu118.sh`,
  `tools/setup_venv_macos_inference.sh`
- UI/API checks: `tools/run_ui_*`, `tools/check_ui_endpoints.py`
- calibration and EDR: `tools/build_ensemble_features.py`,
  `tools/train_ensemble_xgb.py`, `tools/tune_ensemble_thresholds_xgb.py`,
  `tools/run_canonical_prepass_discovery.py`
- automatic-labeling and Falcon diagnostics: `tools/run_auto_label_benchmark.py`,
  `tools/debug_falcon_*.py`
- dataset utilities: `tools/reorder_labelmap.py`,
  `tools/detect_missclassifications.py`

See [tools/README.md](tools/README.md) for a shorter command index.

## Tests and Validation

Fast targeted checks:

```bash
python -m py_compile localinferenceapi.py services/*.py utils/*.py models/*.py
python -m pytest tests/test_api_route_uniqueness.py tests/test_dataset_zip_upload_security.py -q
python -m pytest tests/test_macos_acceleration.py tests/test_detector_sahi_resilience.py -q
```

Broader suites are organized by feature area under `tests/`. UI tests live in
`tests/ui/e2e/` and are marked so they can be run separately when Playwright and
the backend are available.

<details>
<summary>Update tracking</summary>

Latest full backend validation: `1086 passed, 17 skipped`.
Detailed maintenance notes live in
[docs/backend_storage_hardening_log.md](docs/backend_storage_hardening_log.md).

</details>

## Runtime Artifacts

Most generated files are intentionally under `uploads/` and are git-ignored:

- uploaded and linked dataset metadata
- annotation overlays and lock/session state
- training job records and model artifacts
- SAM3/Qwen/agent-mining caches
- calibration jobs and calibration cache
- EDR packages and exported recipes

Do not treat `uploads/` as source code. Back it up separately if a dataset or
trained model matters.

## Third-Party Licenses

Review upstream licenses before using or redistributing model outputs or
artifacts:

- Ultralytics YOLOv8: AGPL-3.0 or commercial terms
- RF-DETR: Apache-2.0
- Meta Segment Anything and SAM3: review repository and model-card terms
- Qwen/Qwen3-VL: review model license and acceptable-use terms
- Falcon Perception: review model-card and remote-code terms
- XGBoost: Apache-2.0

Tator is local tooling; license compliance for models, datasets, and generated
artifacts remains the operator's responsibility.
