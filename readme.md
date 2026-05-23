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

## What Tator Looks Like In Use

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

## Main Workflows

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

## Update Tracking

### 2026-05-19: Qwen captioning and glossary stabilization

This checkpoint focuses on making Qwen captions more inspectable, more
dataset-aware, and less likely to turn label glossary examples into unsupported
object claims.

- Added editable caption glossaries to the labeling flow. The frontend now
  populates glossary defaults from the active dataset labelmap, keeps them
  editable in the Qwen caption panel, and sends the normalized glossary with
  caption requests.
- Tightened glossary semantics in the caption prompts. Glossary entries are
  now framed as broad class meanings plus possible variants, not as a list of
  words the model should force into the caption. For example, `LightVehicle`
  should normally become a broad term such as "small vehicle" unless the image
  clearly supports a subtype.
- Changed windowed captioning defaults so labeled runs focus on windows that
  intersect label hints instead of captioning every crop by default. Empty
  windows can still be enabled explicitly, but they are treated as broad scene
  context rather than as an object inventory.
- Added overlap and deduplication guidance for windowed captions. Boxes that
  cross overlapping crops are clipped into each local crop, keep internal
  source identity for backend grouping, and are described to the model without
  exposing bbox IDs in prompts or outputs.
- Added spatial grounding for windowed caption observations. Full-image
  composition and the window merge pass now receive each crop's approximate
  full-image region, percent bounds, and explicit guidance to translate
  crop-local spatial language into global image language.
- Split the caption workflow explanation in the UI so users can distinguish
  full-image draft/refine passes, window observations, window merge, quality
  guards, model selection, decode mode, and max-token behavior.
- Improved progress visibility for Qwen generation. Caption runs report the
  active step, model, token progress, live generated text, and active crop
  overlay so long downloads or multi-window runs are not silent.
- Added deterministic post-processing guards for repeated, incomplete,
  meta-text, count-conflicting, and non-English outputs. The English rewrite
  guard now only fires for non-Latin script text, not normal punctuation such
  as non-breaking hyphens.
- Added a final glossary subtype stabilizer. If the draft and windows disagree
  between a broad term and a subtype, the final caption is demoted back to the
  broad term instead of letting a later cleanup pass invent specificity such as
  "pickup truck".
- Updated the Qwen caption UI defaults to prefer broad natural class terms, and
  to auto-upgrade the previous default system prompt when the user has not
  customized it. Already-open browser tabs should be hard-reloaded once to pick
  up the new static JavaScript defaults.
- Added an editable caption prompt stack to the Qwen caption UI. Operators can
  expand and edit the user-request, system, detection-context, window,
  draft/refine, window-merge, and cleanup/guard prompt families; normal frontend
  caption runs send those prompt pieces explicitly instead of relying on hidden
  backend wording.
- Aligned the Caption workflow grid with the backend execution path. Each card
  now names the prompt layers, runtime inputs, previous model outputs, and next
  output for that step, and clarifies that Draft + refine is one full-image
  Thinking-model pair rather than a per-crop loop.
- Tightened authoritative count handling. Later cleanup, merge, and coverage
  passes now receive the count inventory, old "do not mention counts" wording
  was narrowed to "do not say counts were provided," and the guard now refines
  outputs that mention a class without its required numeric count.
- Latest editable prompt-stack validation used:
  `python -m py_compile localinferenceapi.py services/qwen.py models/schemas.py`,
  `node --check ybat-master/ybat.js`, `git diff --check`, and
  `pytest tests/test_qwen_caption_prompt.py -q` with 42 passing tests.
- Validation for this checkpoint used:
  `python -m py_compile localinferenceapi.py services/qwen.py utils/glossary.py models/schemas.py api/qwen_caption.py`,
  `node --check ybat-master/ybat.js`, `git diff --check`, and
  `pytest tests/test_qwen_caption_prompt.py tests/test_qwen_progress.py tests/test_qwen_mlx_runtime.py tests/test_dataset_linked_annotation_flows.py -q`
  with 81 passing tests.

### 2026-05-19: SAM3 text windowing and labeling-panel cleanup

This checkpoint turns the SAM3 text prompt engine into a stronger batch tool
for large images and makes labelmap extension explicit from the labeling UI.

- Added SAM3 text windowed mode. The backend slices large images into
  overlapping windows, runs SAM3 text prompting per crop, reprojects boxes and
  masks into full-image coordinates, and fuses duplicates before returning
  detections to the labeling canvas.
- Added SAM3 text window settings to the UI: windowed mode, window size, and
  overlap. Cascade steps and "next N images" batch runs inherit the same
  windowing settings so the behavior is consistent across single-image and
  batch usage. SAM3 polygon creation now honors the per-request simplify
  epsilon from the SAM3 text prompt before falling back to the global polygon
  slider.
- Added labelmap extension from the SAM3 text prompt panel. Operators can add a
  new class, select it for SAM3 output, save the new labelmap, and see the
  retraining warning when an auto-class predictor may have been trained against
  the old class list.
- Persisted saved labelmaps through linked and transient annotation metadata
  endpoints. Backend validation rejects empty, duplicate, or multiline class
  names before writing `labelmap.txt`.
- Collapsed Qwen 3 detection, Qwen 3 captioning, EDR [wip], and SAM3 text
  prompt into separate closed panels by default. YOLO bbox import and "Save
  YOLO + captions" controls now live in the main annotation source panel.
- Fixed batch navigation for SAM3 text/cascade runs across next N images and
  removed a duplicate bbox-folder file-label registration introduced by the UI
  move.
- Fixed singleton-channel SAM mask handling so shared mask encoding, window
  mask reprojection, and mask bbox extraction correctly accept both `(H, W)`
  and `(1, H, W)` masks, and mask-derived boxes now use exclusive right/bottom
  bounds instead of dropping single-pixel-wide masks.
- Aligned windowed SAM3 text result-limit handling with the direct SAM3 text
  path: non-positive API limits are treated as unlimited instead of one result.
- Fixed the UI control-manifest pytest wrapper to invoke the checker from the
  repo root with the active Python interpreter instead of a bare `python`
  executable.
- Added a labeling-panel layout contract test for the closed-by-default
  pop-down panels, EDR tooltip copy, and moved YOLO import/export controls.
- Review validation used:
  `node --check ybat-master/ybat.js`,
  `./.venv-macos/bin/python -m py_compile localinferenceapi.py models/schemas.py services/qwen.py utils/image.py utils/coco.py utils/coords.py services/detector_params.py tests/ui/e2e/test_control_manifest_contract.py tests/test_labeling_panel_layout_contract.py`,
  `git diff --check`,
  and `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_prompt.py tests/test_sam3_text_windowed_prompt.py tests/test_coords_window_normalization.py tests/test_qwen_agentic_coords.py tests/test_dataset_linked_annotation_flows.py tests/test_labeling_panel_layout_contract.py tests/ui/e2e/test_control_manifest_contract.py -q`
  with 78 passing tests.

### 2026-05-20: Caption panel usability and image navigation

This checkpoint tightens the Qwen captioning panel ergonomics and adds a
workspace-focused image navigation shortcut.

- Moved the generated-caption label above the caption output box and made that
  output area three times taller by default.
- Added tooltips to the caption style selector, random style button, style and
  opening prompt editors, inspiration toggles, and caption prompt stack.
- Made caption style/opening prompt boxes and prompt-stack textareas full width,
  and doubled their default vertical space so long prompts are inspectable.
- Added broad Label Images tab keyboard navigation: `Tab` moves to the previous
  image and `Space` moves to the next image. Text-editing fields are still
  protected so prompt/caption editing keeps normal typing behavior.
- Cleaned up the in-app Label Images shortcut list so current shortcuts are
  grouped by image navigation, focus mode, class cycling, selection, box
  editing, SAM, region detect/export, canvas controls, and mode toggles.
- Extended the labeling-panel layout contract test to cover the caption output
  layout, caption prompt controls, guarded keyboard image-navigation shortcuts,
  and the grouped shortcut documentation shown in the UI.
- Validation used `node --check ybat-master/ybat.js`,
  `./.venv-macos/bin/python -m pytest tests/test_labeling_panel_layout_contract.py -q`,
  and `git diff --check`, with 5 passing layout-contract tests.

### 2026-05-20: Class Split Explorer MVP

This checkpoint adds the first pass of per-class and all-class embedding
analysis for open annotation datasets.

- Added a top-level **Class Split Explorer** tab so the embedding graph has a
  full workspace instead of living in the Label Images sidebar. It runs against
  the dataset currently open in the labeling workspace.
- Added `/class_analysis/*` backend jobs. The backend crops current annotation
  objects, embeds them with CLIP or DINOv3, projects the embedding space with
  PCA or optional UMAP, clusters objects, and writes thumbnails plus result
  metadata under `uploads/class_analysis/`.
- Added a local vendored Plotly bundle for the interactive graph. The graph
  supports zoom, pan, hover, lasso/select, class/cluster/outlier/suspicion
  coloring, and class filtering.
- Added all-classes wrong-class review scoring. Objects whose nearest neighbors
  mostly belong to a different class are listed as likely wrong-class
  candidates for inspection.
- Added source-image jump and single-object relabel. Selecting a point shows
  the crop; **See instance** switches back to Label Images, opens the source
  image, and selects the matching bbox. The inspector can also change that
  object to an existing class, save the annotation snapshot, and rerun the same
  analysis.
- Removed the default sample cap. Blank or missing sample cap now means analyze
  every object in scope; entering a positive cap still enables faster sampled
  audits.
- Added review hardening so analysis and relabel reruns refuse to proceed if
  the current annotation snapshot cannot be saved, preventing stale backend
  labels from being embedded after local edits.
- Added a WALDO v4 embedding experiment harness and first benchmark report
  under `docs/waldo_embedding_experiment_plan.md` and
  `uploads/class_analysis/experiments/waldo_v4/`. The minimum matrix showed
  that canonical fixed-size crops alone still leaked object size, while
  canonical `336x336` padded-square crops plus size-bias residualization cut
  the all-class PCA size-axis correlation from about `0.83` to `0.03` without
  materially changing nearest-neighbor class purity.
- Added the remaining-lever and finalist preset experiment matrices under
  `uploads/class_analysis/experiments/waldo_v4_remaining_smoke/` and
  `uploads/class_analysis/experiments/waldo_v4_finalists/`. The screen rejected
  raw/native embeddings and whitening, kept PCA as the diagnostic projection,
  and confirmed three user-facing quality presets: **Fast** (`224px`, padding
  `0.04`), **Balanced** (`336px`, padding `0.08`), and **Precise** (Balanced
  plus tight/context multi-view embeddings).
- Added shared embedding recipe helpers for background suppression and
  multi-view crop composition. Class Split Explorer and auto-class training now
  use the same canonical crop/view metadata, and sampled Class Split runs defer
  expensive crop materialization until after the sampled records are selected.
- Removed the earlier fixed-projection SALAD-style crop aggregator from Class
  Split and auto-class. The replacement SALAD path is stricter: Tator now trains
  its own local SALAD optimal-transport head from user media, stores it in the
  local head registry, and reuses that exact selected head for Data Ingestion
  reference-profile scoring. No upstream or third-party SALAD checkpoint is
  loaded.
- The shared recipe is intentionally narrow in the normal UI: object crops are
  padded to a square, resized to a canonical input size, optionally rendered as
  tight/context views, embedded with the selected CLIP or DINOv3 backbone, and
  then passed through the same size-bias residualizer used by the trained
  auto-class head. Raw/native embeddings and no-adjustment variants stay in the
  experiment harness so normal audits and auto-class inference use the proven
  path.
- Added Class Split quality presets plus an advanced disclosure for crop
  padding, canonical size, DINO pooling, background mode, embedding views, and
  UMAP neighbors. Raw/native embeddings, whitening, and image-bias removal
  remain experiment-harness levers rather than normal UI controls.
- Reran the uncapped finalist recipe suite on the active WALDO v4 snapshot and
  aligned Class Split Explorer with the measured winner. **Precise best**
  (`336px`, padding `0.08`, tight+context views, size-bias residualization,
  PCA) is now the Class Split default because it had the best all-class
  neighbor purity (`0.897` object-weighted, `0.692` class-balanced) while
  keeping absolute size leakage low (`0.027`). **Balanced** remains the faster
  single-view recipe and the conservative auto-class training default. Optional
  UMAP now defaults to `50` neighbors to match the evaluated finalist setup,
  but PCA remains the diagnostic default because UMAP projections can re-expose
  strong size-correlated axes even when the underlying embedding recipe is
  sound.
- Hardened the experiment harness report so it writes explicit absolute
  size-axis leakage, class-balanced all-class nearest-neighbor purity, worst
  class purity, sample-cap metadata, and an all-class ranking. This prevents
  negative signed size correlations from looking better than equally large
  positive size leakage in generated reports.
- Aligned trained auto-class predictors with the Class Split recipe. Training
  now saves crop mode, crop padding, canonical size, DINO pooling, and the
  fitted size-bias residualizer; inference reloads those settings and applies
  the same crop/preprocess/embedding adjustment before scoring. SAM-generated
  boxes now use the same full-image bbox crop path instead of classifying raw
  mask crops directly.
- Hardened the embedding cache and training loop. Class Split cache keys include
  the image hash, bbox, crop recipe, encoder, and pooling mode; cached embedding
  files are now validated for one-dimensional finite vectors before a crop can
  be treated as reusable, so corrupt cache entries force real crop
  rematerialization instead of placeholder encoding. Auto-class training also
  closes source images and generated crop views on both success and failure, and
  DINOv3 training metadata now records the actual encoder model used by the run.
- Validation used `node --check ybat-master/ybat.js`,
  `NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python -m py_compile api/class_analysis.py tools/run_class_split_experiments.py localinferenceapi.py tools/clip_training.py services/classifier.py utils/embedding_recipe.py tests/test_class_analysis.py tests/test_labeling_panel_layout_contract.py tests/test_sam3_text_windowed_prompt.py`,
  `NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python -m pytest tests/test_class_analysis.py tests/test_labeling_panel_layout_contract.py tests/test_sam3_text_windowed_prompt.py -q`
  (44 passing tests), and `git diff --check`.

### 2026-05-21: Data Ingestion and Local SALAD

This checkpoint adds a first-class **Data Ingestion** workspace for deciding
which new images or video frames are worth adding relative to an accepted
reference dataset.

- Added a top-level **Data Ingestion** tab with a reference-first flow. The user
  chooses either the current Label Images dataset or a registered backend
  dataset, builds/selects a local SALAD reference profile from that accepted
  set, and only then submits candidate media for novelty/diversity review.
- Candidate images/videos are embedded with the selected reference profile and
  ranked by greedy farthest-first diversity against the reference set so a user
  can keep, for example, only the top `20%` most novel media.
- Added video ingestion through `ffmpeg`: videos are sampled into frames using
  the configured frame interval and per-video frame cap before embedding.
- Added **local SALAD reference profiles**. Local SALAD heads are initialized
  and trained inside Tator from the user's own reference dataset images with a
  frozen spatial-token encoder underneath. DINOv3 is the default base encoder;
  C-RADIOv4 can be selected as a candidate base encoder. Training progress uses
  plain reference-profile stages: **Preparing reference media**,
  **Encoding reference views**, **Training reference profile**,
  **Optimizing reference profile**, and **Finalizing reference profile**.
- Added **Build reference profile** in Data Ingestion. Active Label Images
  references are uploaded from the browser; backend datasets are resolved
  server-side without re-uploading. The max-reference cap defaults to `0`,
  meaning every reference image/frame is used unless the user explicitly sets a
  smaller safety cap.
- Enforced a local-only SALAD policy. Tator does not load upstream SALAD
  checkpoints; saved heads must use the Tator `local-salad-v1` format and the
  `local_training_only_no_external_salad_checkpoint` policy marker before they
  can be selected. New heads also carry a `tator_local_salad_trainer` marker so
  generic external `.pt` files are rejected even if they are dropped into the
  local folder.
- Added `/data_ingestion/*` backend jobs for capabilities, local SALAD training,
  diversity analysis, result retrieval, and cancellation. Results are written
  under `uploads/data_ingestion/`; local heads are stored under
  `uploads/salad_heads/`.
- Kept local SALAD out of Class Split and auto-class UI presets after the crop
  benchmark showed no promotion signal. DINOv3 remains the established
  crop-level baseline; C-RADIOv4 remains available for reference-profile
  comparison. Class Split, auto-class training, and auto-class inference now
  reject manual `embedding_aggregation=local_salad` requests/artifacts so local
  SALAD remains scoped to Data Ingestion reference-profile scoring.
- Removed dormant Class Split and auto-class frontend request wiring for local
  SALAD controls after those controls were removed from the UI, and suppress
  shift-wheel events over the Class Split plot so the abandoned pan shortcut
  cannot trigger jittery Plotly interactions.
- Tightened the Data Ingestion profile picker so it only offers local SALAD
  reference profiles that match the currently selected active/backend reference
  dataset, instead of auto-selecting an unrelated global head and later
  disabling candidate analysis.
- Made the same reference/profile check fail closed in the backend. Local SALAD
  analysis now rejects heads with missing reference metadata or a mismatched
  `reference_dataset_id`, so direct API calls cannot bypass the Data Ingestion
  reference-first flow. The Data Ingestion analysis job path now rejects pooled
  DINOv3/C-RADIO encoder requests as unsupported; pooled encoders remain
  internal helpers/benchmark paths, not ingestion analysis modes.
- Added `tools/benchmark_salad_diversity.py` to compare pooled DINOv3 selection
  against a locally trained SALAD head:

```bash
NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/benchmark_salad_diversity.py \
  --image-dir uploads/datasets/labeling_session_1/train/images \
  --sample-cap 8 --keep-fraction 0.25 --train-local-salad --epochs 1 --batch-size 2 \
  --delete-trained-head --output-dir uploads/data_ingestion/benchmarks/smoke
```
- Validation used `node --check ybat-master/ybat.js`, `git diff --check`,
  `NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python -m py_compile
  localinferenceapi.py api/data_ingestion.py services/data_ingestion.py
  utils/local_salad.py tools/benchmark_salad_diversity.py
  tools/clip_training.py tools/train_clip_regression_from_YOLO.py
  tools/run_class_split_experiments.py services/classifier.py
  utils/embedding_recipe.py`, and
  `NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python -m pytest
  tests/test_data_ingestion.py tests/test_class_analysis.py
  tests/test_labeling_panel_layout_contract.py -q` (47 passing tests). A
  16-image smoke benchmark trained a throwaway local SALAD head and compared it
  against pooled DINOv3; the two selectors overlapped on two of four kept items
  (`Jaccard=0.333`), confirming the local train/load/analyze path without
  keeping the generated head.
- Added `tools/benchmark_salad_class_separation.py` as a benchmark-only check
  of pooled crop recipes against local SALAD crop-token aggregation before
  deciding whether SALAD should be promoted into Class Split or auto-class:

```bash
NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/benchmark_salad_class_separation.py \
  --dataset-root uploads/datasets/labeling_session_1 \
  --image-cap 64 --object-sample-cap 128 --train-local-salad --epochs 1 \
  --batch-size 4 --delete-trained-head
```
- Current smoke result on `labeling_session_1` shows local SALAD trains and
  runs correctly, but a one-epoch 32-image head does **not** beat pooled DINOv3
  for object class separation yet: Balanced pooled was ranked first
  (`class-balanced NN purity 0.240`, size-axis abs corr `0.039`), Precise pooled
  second (`0.232`, `0.046`), and local SALAD third (`0.185`, `0.050`). The UI
  therefore keeps local SALAD in Data Ingestion only and does not expose it as a
  crop-level Class Split or auto-class preset.

### 2026-05-21: C-RADIOv4 Candidate Embeddings

This checkpoint adds NVIDIA C-RADIOv4 as a first-class candidate encoder for
embedding-derived workflows while keeping DINOv3 as the default until larger
dataset benchmarks justify a switch.

- Added shared C-RADIOv4 helpers for `nvidia/C-RADIOv4-SO400M` and
  `nvidia/C-RADIOv4-H`. The backend uses Hugging Face Transformers remote-code
  loading and the model's `summary` plus spatial token outputs.
- Added C-RADIOv4 pooling modes: `summary`, `spatial_mean`, and
  `summary_spatial_concat`. These are exposed in Class Split, auto-class
  training metadata, the C-RADIO experiment matrix, and local SALAD
  reference-profile training.
- Added C-RADIOv4 to Class Split Explorer, Data Ingestion, local SALAD
  reference-profile training/scoring, and auto-class training/inference
  metadata. Local SALAD heads now record their base encoder (`dinov3` or
  `cradio`), encoder model, and C-RADIO pooling mode; Data Ingestion scoring
  rejects a head if the saved encoder type does not match the requested profile
  path.
- C-RADIOv4 local SALAD heads are trained over the spatial-token channel width.
  If a model exposes a wider global `summary` vector than its spatial tokens,
  the local SALAD head uses the spatial-token mean for the global descriptor so
  training and inference stay width-consistent.
- Added `open_clip_torch>=3.3,<4.0` to `requirements.txt` and
  `requirements-macos-inference.txt` because NVIDIA's C-RADIOv4 remote-code
  model imports `open_clip`.
- Acceleration status: CUDA still works through Torch when available. On Mac,
  `auto` now prefers the local `~/cradio_mlx` runtime and its
  `checkpoints/c-radiov4-so400m` or `checkpoints/c-radiov4-h` safetensors
  checkpoints. If that local framework, MLX, or the requested checkpoint is not
  present, C-RADIOv4 falls back to Torch CUDA/MPS/CPU. The backend reports the
  resolved path in capabilities so the UI can distinguish local MLX from Torch
  fallback. Local SALAD now has its own MLX backend, so Mac auto mode can keep
  both C-RADIO token extraction and SALAD aggregation/training in MLX when
  available.
- Added benchmark levers:

```bash
NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root <dataset-root> --label-zip <labels.zip> --labelmap <labelmap.txt> \
  --image-dir <images> --matrix cradio --sample-cap 128

NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/benchmark_salad_diversity.py \
  --image-dir uploads/datasets/labeling_session_1/train/images \
  --sample-cap 64 --keep-fraction 0.2 --include-cradio-pooled

NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/benchmark_salad_class_separation.py \
  --dataset-root uploads/datasets/labeling_session_1 \
  --image-cap 64 --object-sample-cap 128 --include-cradio
```

- Smoke validation completed after installing the new dependency into
  `.venv-macos`. A two-image Data Ingestion smoke wrote
  `uploads/data_ingestion/benchmarks/cradio_audit/20260521_143717/benchmark.json`
  and compared DINOv3 pooled with C-RADIOv4 pooled. An eight-object Class Split
  smoke wrote
  `uploads/class_analysis/benchmarks/cradio_audit/20260521_143747/benchmark.json`
  and proved that C-RADIOv4 crop embeddings run through the same size-bias,
  projection, scoring, and recommendation path.
- A two-image, one-epoch C-RADIO local SALAD smoke wrote
  `uploads/data_ingestion/benchmarks/cradio_salad_audit/20260521_144151/benchmark.json`.
  It trained a throwaway C-RADIO-backed local SALAD head, used it for Data
  Ingestion diversity scoring, emitted a Class Split/auto-class aggregation
  recommendation, and deleted the test head afterward.
- A synthetic YOLO auto-class smoke trained a C-RADIOv4 logistic-regression head
  through `tools/train_clip_regression_from_YOLO.py`, wrote
  `/tmp/tator_cradio_autoclass_smoke/cradio_logreg.pkl`, and verified the saved
  metadata reloads as `encoder_type=cradio`, `encoder_model=nvidia/C-RADIOv4-SO400M`,
  `cradio_pooling=summary`. The CLI also now runs correctly as a direct script
  and handles scikit-learn versions that removed the deprecated
  `LogisticRegression(multi_class=...)` constructor argument.
- Full uncapped WALDO C-RADIO matrix completed on 2026-05-21:

```bash
NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root uploads/datasets/labeling_session_1 \
  --label-zip /tmp/tator_waldo_labels.zip \
  --labelmap uploads/datasets/labeling_session_1/labelmap.txt \
  --image-dir uploads/datasets/labeling_session_1/train/images \
  --matrix cradio --sample-cap 0 \
  --output-root uploads/class_analysis/benchmarks/waldo_cradio_full
```

  Artifacts: `uploads/class_analysis/benchmarks/waldo_cradio_full/`.

| C-RADIO recipe | Projection | Object-weighted NN purity | Class-balanced NN purity | Abs size leakage | Wrong-class candidates | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Summary | PCA | `0.8876` | `0.7006` | `0.0663` | `1025` | `4893.5s` |
| Summary | UMAP | `0.8876` | `0.7006` | `0.6722` | `1025` | `81.1s` |
| Spatial mean | PCA | `0.8898` | `0.7001` | `0.0672` | `964` | `5006.4s` |
| Summary + spatial mean | PCA | `0.8912` | `0.7053` | `0.0611` | `983` | `5022.3s` |
| Precise tight + context | PCA | `0.9078` | `0.7264` | `0.0591` | `851` | `9046.9s` |

- Current recommendation remains conservative: DINOv3 Precise stays the default
  for Class Split, while Data Ingestion now uses a selected local SALAD
  reference profile trained from the accepted dataset.
  C-RADIOv4 tight+context audit improved nearest-neighbor purity on the full
  WALDO run (`0.9078` object-weighted / `0.7264` class-balanced versus DINOv3
  Precise at `0.8969` / `0.6917`), but it was about 200x slower than DINOv3
  Precise on this Mac (`9046.9s` versus `45.3s`), had higher size-axis leakage
  (`0.0591` versus `0.0273`), and produced more wrong-class candidates (`851`
  versus `804`). Treat C-RADIOv4 as an opt-in slow audit path, not the default.
- The UI now exposes the benchmark context in Class Split Explorer and Class
  Predictor Settings. The note explains that nearest-neighbor
  purity means same-class agreement among nearby object-crop embeddings, reports
  the DINOv3 Precise and C-RADIOv4 tight+context audit numbers side by side,
  warns that the C-RADIO number is the slow audit path rather than the default
  summary preset, and explicitly states that no full uncapped WALDO promotion row
  is recorded for CLIP in this benchmark set.
- Backlog: improve C-RADIO cache handling so one forward pass can cache the raw
  summary plus spatial-token outputs and derive `summary`, `spatial_mean`, and
  `summary_spatial_concat` embeddings from that shared record. The current
  per-pooling embedding cache can recompute the same C-RADIO forward for
  adjacent benchmark variants.

### 2026-05-22: C-RADIOv4 Local MLX Backend

- Wired the local `~/cradio_mlx` framework into Tator's shared C-RADIO helper.
  On macOS, `CRADIO_BACKEND=auto` now resolves to `mlx` when the local package
  and requested checkpoint are present. Existing consumers keep calling
  `load_cradio_backbone()` and `encode_cradio_images()`, so Class Split, Data
  Ingestion, local SALAD C-RADIO training/scoring, and C-RADIO auto-class
  training/inference use the same backend selection.
- Default checkpoint discovery:
  - `nvidia/C-RADIOv4-SO400M` -> `~/cradio_mlx/checkpoints/c-radiov4-so400m`
  - `nvidia/C-RADIOv4-H` -> `~/cradio_mlx/checkpoints/c-radiov4-h`
- Useful overrides: `CRADIO_MLX_ROOT`, `CRADIO_MLX_SRC`,
  `CRADIO_MLX_SO400M_CHECKPOINT`, `CRADIO_MLX_H_CHECKPOINT`,
  `CRADIO_MLX_CHECKPOINT`, `CRADIO_MLX_DTYPE`, `CRADIO_MLX_IMAGE_SIZE`, and
  `CRADIO_MLX_PRESERVE_INPUT_SIZE=1`. The default MLX input size remains 512px
  to avoid accidentally sending large raw ingestion images through the ViT; use
  `CRADIO_MLX_IMAGE_SIZE` for controlled resolution experiments. If MLX import
  or checkpoint discovery fails, auto mode falls back to the existing Torch
  CUDA/MPS/CPU path and reports the fallback in backend capabilities.

### 2026-05-22: Local SALAD MLX Runtime

- Added a Tator-owned MLX implementation of the existing local SALAD head in
  `utils/local_salad_mlx.py`. It mirrors the current locally trained
  optimal-transport aggregation head: token/global projection MLPs, learned
  dustbin score, Sinkhorn assignment, cluster pooling, and symmetric InfoNCE
  training.
- `LOCAL_SALAD_BACKEND=auto` is now the standard Mac behavior. On macOS it
  resolves to `mlx` when MLX is installed; otherwise it falls back to the
  existing Torch implementation. `LOCAL_SALAD_BACKEND=torch` and
  `LOCAL_SALAD_BACKEND=mlx` can force either path.
- The saved head contract stays compatible. MLX-trained heads are serialized
  back into the same Tator `.pt` payload with the same state-dict keys, plus
  metadata such as `salad_backend=mlx`. Existing Torch-trained heads can be
  loaded into the MLX runtime by mapping their state dict into MLX arrays.
- Class Split, Data Ingestion, local SALAD diversity scoring, and C-RADIO
  auto-class paths now use the same backend resolver. SALAD-MLX remains scoped
  to the local SALAD Data Ingestion profile/scoring path while preserving Torch
  fallback for DINOv3/Torch and non-Mac environments.

### 2026-05-23: Class Comparison and Auto-Class Debug Closure

- Cleaned up the Class Split / Class Analysis UI contract after the crop-review
  pass. The selected-crop assigner stays at the top of the right-side stack,
  crop previews scale to the available viewer area, hover tooltips no longer
  inject file-input artifacts into the plot, and the likely-wrong-class review
  panel is collapsible. Shift-scroll panning was removed from the plot instead
  of keeping the jittery shortcut.
- Kept the SALAD boundary explicit. Class comparison capabilities now expose
  pooled recipes only (`balanced`, `precise`, and `cradio_summary`) with
  `embedding_aggregation_modes=["pooled"]`; local SALAD remains in Data
  Ingestion reference-profile scoring and training, not in crop-level Class
  Split or auto-class presets.
- Fixed binary logistic-regression replay for trained auto-class heads. Runtime
  inference now applies saved logit adjustment before calibration temperature,
  and training metrics / hard-mining predictions use the same adjusted
  probability path instead of mixing raw `predict()` labels with adjusted
  probabilities.
- Fixed retraining of the currently active classifier. When a training job
  overwrites the active classifier artifact, the backend refreshes the in-memory
  classifier, labelmap, metadata, active encoder, and normalized runtime head.
  Active C-RADIO classifiers also resume through the C-RADIO backbone path
  rather than the CLIP/DINO-only reload path.
- Hardened active classifier activation. MLP heads now fail closed when layer
  input widths, layer-norm widths, or final output width do not match the saved
  class list. `set_active_model` validates the normalized runtime head before
  mutating active globals, so a malformed classifier cannot replace a working
  active model with a stale or unavailable head.
- Hardened CLIP backbone replay for explicit classifier heads. DINOv3 and
  C-RADIO already reload the encoder recorded in the classifier metadata; CLIP
  scoring now does the same by using the saved `encoder_model` / `clip_model`
  for non-active classifier IDs without mutating the active CLIP model. The
  post-training resume path also reloads the active CLIP model name instead of
  falling back to a stale pre-training backbone.
- Fixed active classifier propagation into deep prepass cleanup. The prepass
  runner now resolves the active classifier head lazily when cleanup runs,
  instead of capturing the import-time head, so active-model changes and
  active-classifier retraining refreshes are honored by auto-label filtering.
- Fixed stale active-classifier cleanup. Deleting the active classifier, or
  discovering that its backing file disappeared, now clears the loaded
  classifier, labelmap, metadata, and normalized head instead of letting
  inference continue with an unlisted in-memory model. Active-model activation
  also uses strict root-relative path checks for classifiers and labelmaps, so
  sibling directories with matching path prefixes cannot be treated as upload
  roots.
- Fixed auto-class training artifacts after class filtering. The saved
  classifier labelmap now contains the non-background classes that actually
  remain in the trained head, so low-frequency classes filtered out by
  `min_per_class` cannot make the newly trained model fail active-model
  activation. Train/test splitting also falls back from grouped validation when
  a grouped split would leave a class only in validation, and the CLI confusion
  matrix now reports labels that match the matrix rows/columns.
- Tightened the remaining filtered-class edge cases. Training artifacts now
  report trained positive classes separately from raw encountered classes,
  cached embedding runs recompute YOLO raw class counts before applying
  `min_per_class`, and tiny rare-class splits can carry an empty validation set
  without crashing on a zero-row memmap.
- Validation used:

```bash
.venv-macos/bin/pytest -q tests/test_class_analysis.py
.venv-macos/bin/pytest -q \
  tests/test_class_analysis.py \
  tests/test_clip_training_artifact_publish.py \
  tests/test_train_clip_regression_cli.py \
  tests/test_classifier_batching.py \
  tests/test_classifier_infer_clip_model_signature.py \
  tests/test_clip_model_infer_dim.py
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python -m pytest tests --ignore=tests/ui/e2e -q
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python -m py_compile localinferenceapi.py services/classifier.py tools/clip_training.py tools/train_clip_regression_from_YOLO.py
node --check ybat-master/ybat.js
git diff --check
```

- Latest local backend smoke after restart:

```bash
curl -sS http://127.0.0.1:8000/clip/active_model
curl -sS http://127.0.0.1:8000/class_analysis/capabilities
```

## Training and Model Management

Tator keeps helper models close to the annotation workflow:

- **CLIP/DINOv3 heads**: fast class predictors trained from managed datasets.
- **YOLOv8**: detector training, active run selection, run summaries, downloads,
  deletion, and experimental head grafting for disjoint new classes.
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
CLIP/SAM/SAM3/detectors and MLX-VLM for Qwen when available. Full detector/SAM
training remains Linux/CUDA-first, but Qwen MLX LoRA adapter jobs are available
for small enough local Apple Silicon models.

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
