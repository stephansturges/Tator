# WALDO Class Split Embedding Experiment Plan

Date: 2026-05-20

## Objective

Find the best Class Split Explorer embedding setup for the active WALDO v4 dataset by testing the remaining embedding levers systematically, using measurable graph quality, size-bias diagnostics, wrong-class discovery quality, and human-inspection checkpoints.

## Dataset Lock

The requested path `~/Pictures/WALDO/WALDO_new_date_for_v4` was not present locally. The matching available dataset is:

`/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4`

Use this dataset snapshot unless the user supplies a different path.

Inputs:

- Images: `/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/cropped images`
- Labelmap: `/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/code/labelmap.txt`
- Labels: `/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/labels/bboxes_yolo (207).zip`

Latest label zip evidence:

- `bboxes_yolo (207).zip`
- Modified: `2026-05-20 14:05:51`
- Size: `1.6M`

Dataset audit:

- Image files: `3721`
- Label files in latest zip: `3090`
- Non-empty label files: `2253`
- Empty label files: `837`
- Total objects: `31278`
- Missing images for label files: `0`
- Malformed label lines: `0`
- Label image size: `1000x1000` for all matched label images

Class counts:

| Class | Count | Median bbox area pct | P90 bbox area pct |
| --- | ---: | ---: | ---: |
| LightVehicle | 3636 | 0.3751 | 1.0988 |
| Person | 2478 | 0.0234 | 0.0675 |
| Building | 17572 | 1.1970 | 7.2790 |
| UPole | 3365 | 0.1107 | 0.6834 |
| Boat | 547 | 0.5148 | 1.6836 |
| Bike | 1169 | 0.0300 | 0.1505 |
| Container | 211 | 0.5768 | 1.9796 |
| Truck | 586 | 0.2892 | 0.9212 |
| Gastank | 764 | 0.0811 | 0.4488 |
| Digger | 152 | 0.4027 | 2.0286 |
| Solarpanels | 763 | 0.9676 | 7.9178 |
| Bus | 35 | 0.4329 | 1.1388 |

Implications:

- Building dominates object count and should not be allowed to dominate all-class ranking.
- Person and Bike are extremely small; any setup that works only on larger classes is not good enough.
- Bus and Digger are low-count classes, so they should be evaluated with qualitative review and nearest-neighbor stability rather than only cluster metrics.

## Current Baseline

Use the current Class Split pipeline as baseline:

- Encoder: DINOv3
- Backbone: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Projection: PCA and UMAP
- Projection neighbors: `15`, `50`, `0` for all neighbors
- Crop normalization: `canonical`
- Canonical size: `336`
- Size-bias control: `remove_size_bias`
- Crop padding: `0.08`
- Scoring neighbors: `15`
- Sample cap: none for final runs

Baseline outputs must include:

- `result.json`
- `config.json`
- `embeddings.npz`
- `metadata.jsonl`
- graph report with size-axis correlation, cache stats, class counts, cluster summary

## Metrics

Every run should produce a metrics row with:

- `run_id`
- encoder type and model
- preprocessing mode
- crop padding and object occupancy settings
- background mode
- DINO pooling mode
- embedding aggregation mode
- multi-scale recipe
- embedding adjustment
- whitening or PCA denoise setting
- projection method and projection neighbors
- scoring neighbors
- per-class object count used
- embedding cache hit rate
- runtime
- strongest projection-size correlation
- mean absolute size correlation across x/y and bbox/crop area/aspect features
- KMeans best `k`
- KMeans silhouette with cosine metric
- nearest-neighbor same-class purity for all-class runs
- wrong-class candidate count
- manually accepted wrong-class rate from a reviewed sample
- per-class neighbor purity
- per-class outlier concentration

Primary selection criteria:

1. Low size-axis leakage: strongest absolute size correlation should be below `0.35` where possible.
2. High local semantic purity: same-class nearest-neighbor purity should improve or remain stable.
3. Useful clusters: clusters should separate meaningful subtypes without merely separating object size.
4. Human usefulness: the top 50 outliers or wrong-class candidates should contain enough real issues or meaningful subtype differences to justify the setup.
5. Runtime acceptable on full WALDO: cache makes reruns practical.

## Stage 0 - Build The Experiment Harness

Add a backend or CLI harness that can run Class Split configurations against a fixed dataset snapshot without using the browser.

Required behavior:

- Create a manifest from the WALDO image folder, latest zip, and labelmap.
- Run one class at a time and all classes.
- Write each run under `uploads/class_analysis/experiments/waldo_v4/<run_name>/`.
- Reuse the Class Split crop and embedding cache.
- Emit `metrics.json`, `metrics.csv`, and a small `report.md`.
- Support resumable runs: skip runs with complete artifacts unless `--force`.
- Support `--dry-run` to list planned runs.

Initial command shape:

```bash
./.venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4" \
  --label-zip "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/labels/bboxes_yolo (207).zip" \
  --labelmap "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/code/labelmap.txt" \
  --image-dir "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/cropped images" \
  --output-root uploads/class_analysis/experiments/waldo_v4
```

## Stage 1 - Sanity And Bias Baselines

Run these on all classes and on the priority classes `LightVehicle`, `Person`, `Bike`, `UPole`, `Boat`, `Truck`, `Gastank`, and `Building`.

Matrix:

| ID | Encoder | Backbone | Preprocess | Size bias | Projection |
| --- | --- | --- | --- | --- | --- |
| B1 | DINOv3 | ViT-B LVD | native | none | PCA |
| B2 | DINOv3 | ViT-B LVD | canonical | none | PCA |
| B3 | DINOv3 | ViT-B LVD | canonical | remove_size_bias | PCA |
| B4 | DINOv3 | ViT-B LVD | canonical | remove_size_bias | UMAP n=15 |
| B5 | DINOv3 | ViT-B LVD | canonical | remove_size_bias | UMAP n=50 |
| B6 | DINOv3 | ViT-B LVD | canonical | remove_size_bias | UMAP n=all |

Gate:

- If B3 does not materially reduce size correlation compared with B1/B2, fix residualization before running larger experiments.
- If UMAP n=15/50/all produces similar graph quality, keep n=50 as the default for review because it is usually more stable than very local UMAP.

## Stage 2 - Canonical Crop Geometry

Before pooling experiments, test canonical crop geometry directly.

Canonical crop variants:

- `canonical_224_occ95`: final size 224, object fills most of the canvas.
- `canonical_336_occ85`: current-style fixed crop with moderate context.
- `canonical_448_occ75`: larger input with more context and less object occupancy.
- `canonical_336_occ65`: context-heavy crop for buildings, boats, containers, and solar panels.

Gate:

- Prefer the occupancy setting that lowers size-axis leakage without removing useful subtype/context information.
- If tiny classes need high occupancy and contextual classes need lower occupancy, keep occupancy as a per-run/per-class option instead of forcing one universal value.

## Stage 3 - DINO Pooling Modes

Implement and test DINO pooling modes:

- `pooler`: current default when available
- `cls`: CLS token
- `patch_mean`: mean over patch tokens
- `cls_patch_concat`: L2-normalized concat of CLS and patch mean
- `last4_patch_mean`: average patch means from last four hidden layers if available

Matrix:

- Fixed: canonical crop, remove size bias, padding `0.08`, UMAP n=50, PCA companion run.
- Vary: pooling mode.
- Scope: all classes plus the priority single-class runs.

Gate:

- Prefer the pooling mode that reduces size-axis leakage and improves same-class neighbor purity.
- If `patch_mean` or `cls_patch_concat` is better for small-object classes but worse for Building, consider per-class default suggestions.

## Stage 3b - Token Aggregation Closure

Closed for the original pooled crop-pipeline defaults. The earlier
fixed-projection SALAD-style crop aggregator did not beat the pooled precise
recipe and stays removed.

The replacement SALAD path is local-only and trainable. Tator initializes and
trains its own SALAD optimal-transport aggregation head from user-provided
images/video frames, saves it under `uploads/salad_heads/`, and only reloads
heads carrying the Tator `local-salad-v1`,
`local_training_only_no_external_salad_checkpoint`, and
`tator_local_salad_trainer` markers. No upstream SALAD checkpoint is loaded.
The trained local head is used for Data Ingestion diversity scoring. The
intended default workflow is to choose the accepted reference dataset first
(either the active Label Images dataset or a registered backend dataset), train
and freeze a local SALAD reference profile from that set, and then use that head
to score candidate images/video frames against the accepted reference bank.
The first local Class Split smoke test intentionally keeps pooled DINOv3 as the
default: a one-epoch, 32-image local SALAD head trained and ran cleanly, but
ranked behind pooled Balanced and pooled Precise on class-balanced nearest
neighbor purity. Local SALAD remains a Data Ingestion diversity-scoring path;
crop-level Class Split and auto-class UI presets stay on pooled embeddings.

## Stage 4 - Multi-Scale Object Context

Implement multi-scale crop fusion:

- `tight`: crop around bbox with no extra context.
- `standard`: current padded square.
- `context`: larger crop, e.g. padding `0.25`.
- `tight_standard_concat`: concat tight and standard embeddings.
- `standard_context_concat`: concat standard and context embeddings.
- `weighted_tight_context`: normalized weighted sum, e.g. 70 percent tight, 30 percent context.

Matrix:

- Fixed: best pooling from Stage 3, remove size bias, UMAP n=50.
- Vary: multi-scale recipe.

Gate:

- For tiny classes Person, Bike, Gastank, and UPole, tight or tight-heavy recipes should improve neighbor purity.
- For contextual classes Building, Solarpanels, Container, and Boat, standard/context recipes may be better.
- Do not pick a setup that improves Building by making Person/Bike worse.

## Stage 5 - Background Suppression

Implement background modes:

- `full_crop`: current full crop.
- `mean_fill_outside_box`: keep object rectangle, fill outside it with crop mean.
- `blur_outside_box`: keep object rectangle, blur outside it.
- `darken_outside_box`: keep object rectangle, dim outside it.
- `polygon_mask` when polygon labels are available later.

For bbox-only labels in this WALDO zip, the suppression region is the original bbox inside the padded/canonical crop.

Gate:

- Background suppression should help tiny objects and reduce scene/location clustering.
- Reject modes that destroy useful context for Boat, Building, Solarpanels, or Container.

## Stage 6 - Post-Processing And Denoising

Test embedding transforms:

- L2 only
- remove size bias
- remove size + position bias
- remove size + image-id scene bias
- PCA denoise to 64 dims, then L2
- PCA whiten to 64 dims, then L2
- PCA denoise to 128 dims, then L2
- PCA whiten to 128 dims, then L2

Gate:

- Whitening must improve neighbor purity or reduce size leakage without creating unstable clusters.
- Scene-bias removal is accepted only if it improves all-class wrong-label discovery without flattening real class differences.

## Stage 7 - Backbone Comparison

Test backbones:

- DINOv3 ViT-B LVD
- DINOv3 ViT-L LVD, if local hardware can tolerate it
- DINOv3 SAT backbone, because WALDO is aerial/satellite-like
- C-RADIOv4 SO400M with `summary`, `spatial_mean`, and
  `summary_spatial_concat` pooling
- C-RADIOv4 H if local hardware can tolerate the larger model
- CLIP ViT-L/14

Gate:

- Compare each backbone under the best preprocessing from Stages 2-5.
- SAT should be seriously considered if it improves aerial object neighbor purity.
- C-RADIOv4 should be promoted only if a full class-balanced benchmark beats
  DINOv3 Precise without reintroducing size-axis leakage, runtime blowup, or
  more wrong-class candidates. The 2026-05-21 full WALDO run improved
  nearest-neighbor purity only in the slowest recipe, but did not clear the
  default-promotion gate.
- CLIP can win if semantic grouping improves enough, but reject it if it loses small-object visual distinctions.

C-RADIOv4 implementation note:

- Tator uses the shared C-RADIOv4 helper everywhere C-RADIO embeddings are
  needed. On macOS, `CRADIO_BACKEND=auto` now prefers the local `~/cradio_mlx`
  runtime and matching checkpoint when present; otherwise it falls back to the
  Hugging Face/Transformers path with `open_clip_torch>=3.3,<4.0`, Torch CUDA,
  MPS, or CPU.
- C-RADIO-backed local SALAD heads are trained over the spatial-token channel
  width. When the C-RADIO global summary width differs from the spatial-token
  width, Tator falls back to the spatial-token mean for the SALAD global
  descriptor so training and inference remain shape-consistent.
- Existing WALDO promotion numbers for C-RADIOv4 were gathered on the older
  Torch/MPS path and still do not clear the default-promotion gate. Re-run the
  benchmark matrix on the active hardware before promoting any C-RADIO recipe.
- `tools/run_class_split_experiments.py --matrix cradio` runs the focused
  C-RADIO pooling/backbone screen. `tools/benchmark_salad_diversity.py
  --include-cradio-pooled` and `tools/benchmark_salad_class_separation.py
  --include-cradio` compare C-RADIO against the DINOv3 baselines.

Full WALDO C-RADIO matrix, uncapped, completed on 2026-05-21:

```bash
NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root uploads/datasets/labeling_session_1 \
  --label-zip /tmp/tator_waldo_labels.zip \
  --labelmap uploads/datasets/labeling_session_1/labelmap.txt \
  --image-dir uploads/datasets/labeling_session_1/train/images \
  --matrix cradio --sample-cap 0 \
  --output-root uploads/class_analysis/benchmarks/waldo_cradio_full
```

Artifacts:

- `uploads/class_analysis/benchmarks/waldo_cradio_full/leaderboard.csv`
- `uploads/class_analysis/benchmarks/waldo_cradio_full/report.md`
- `uploads/class_analysis/benchmarks/waldo_cradio_full/metrics.json`

All-class C-RADIO results:

| Recipe | Projection | Object-weighted NN purity | Class-balanced NN purity | Abs size leakage | Wrong-class candidates | Runtime | Cache hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Summary | PCA | `0.8876` | `0.7006` | `0.0663` | `1025` | `4893.5s` | `0.20` |
| Summary | UMAP | `0.8876` | `0.7006` | `0.6722` | `1025` | `81.1s` | `1.00` |
| Spatial mean | PCA | `0.8898` | `0.7001` | `0.0672` | `964` | `5006.4s` | `0.20` |
| Summary + spatial mean | PCA | `0.8912` | `0.7053` | `0.0611` | `983` | `5022.3s` | `0.20` |
| Precise tight + context | PCA | `0.9078` | `0.7264` | `0.0591` | `851` | `9046.9s` | `0.20` |

Decision:

- C-RADIOv4 Precise tight+context is the strongest C-RADIO candidate by
  nearest-neighbor purity and beats DINOv3 Precise there (`0.9078` /
  `0.7264` vs. `0.8969` / `0.6917`).
- DINOv3 Precise remains the default because it is far faster (`45.3s` vs.
  `9046.9s`), has lower size-axis leakage (`0.0273` vs. `0.0591`), and found
  fewer wrong-class candidates (`804` vs. `851`).
- Keep C-RADIOv4 as an opt-in slow audit path until cache sharing and/or a real
  MLX implementation changes the speed profile.

Backlog:

- Improve C-RADIO cache handling so one model forward stores raw summary and
  spatial-token outputs for a crop. Pooling variants should then derive
  `summary`, `spatial_mean`, and `summary_spatial_concat` embeddings from that
  shared cache instead of recomputing the same C-RADIO forward pass across
  adjacent benchmark rows.

## Stage 8 - Ensemble Embeddings

Test late-fusion embeddings:

- DINO-only best
- CLIP-only best
- DINO tight + DINO context
- DINO best + CLIP, equal weight
- DINO best + CLIP, 70/30
- DINO best + CLIP, 85/15

Gate:

- Ensemble wins only if it improves reviewed outlier usefulness and all-class wrong-label discovery.
- If ensemble improves semantic class separation but hides intra-class subtype separation, keep it optional rather than default.

## Stage 9 - Human Review Protocol

For each finalist run:

1. Review the top 50 wrong-class candidates from the all-class graph.
2. Review the top 30 outliers for each priority single-class run.
3. For LightVehicle, manually tag sample clusters as car, van, truck-like, tuk-tuk, pickup-like, ambiguous, or wrong label.
4. For Person, Bike, UPole, and Gastank, check whether clusters are semantic or just size/blur/scene artifacts.
5. Record accept/reject notes into `review.csv`.

Human-facing winner criteria:

- LightVehicle clusters should expose meaningful subtype structure.
- All-class graph should surface plausible wrong-class labels.
- Small-object classes should not collapse into pure crop-size or blur axes.
- The graph should remain usable without excessive manual filtering.

## Final Decision

Select:

- Best default setup for Class Split Explorer.
- Best optional "small object" setup, if different.
- Best optional "semantic class audit" setup, if different.
- Whether UMAP n=50 or n=all should be exposed as the recommended default.
- Whether DINO SAT should become the default for aerial datasets.

Final deliverables:

- `uploads/class_analysis/experiments/waldo_v4/leaderboard.csv`
- `uploads/class_analysis/experiments/waldo_v4/report.md`
- `uploads/class_analysis/experiments/waldo_v4/review.csv`
- Recommended default settings for the UI.
- Any required code changes to make the winning setup available in Class Split Explorer.

## Minimum First Run

Before launching the full matrix, run:

- B1, B2, B3, B5 for LightVehicle.
- B1, B2, B3, B5 for Person.
- B1, B2, B3, B5 for all classes.

This validates the harness, cache, size-bias metrics, and UMAP wiring before spending time on the complete matrix.

## Minimum Run Results

Executed on 2026-05-20 with:

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4" \
  --label-zip "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/labels/bboxes_yolo (207).zip" \
  --labelmap "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/code/labelmap.txt" \
  --image-dir "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/cropped images" \
  --output-root uploads/class_analysis/experiments/waldo_v4
```

Artifacts:

- `uploads/class_analysis/experiments/waldo_v4/leaderboard.csv`
- `uploads/class_analysis/experiments/waldo_v4/report.md`
- `uploads/class_analysis/experiments/waldo_v4/metrics.json`

Key results:

| Run | Scope | Objects | Projection | Size corr | Mean size corr | NN purity |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| B1 | LightVehicle | 3571 | PCA | 0.628 | 0.222 | 1.000 |
| B2 | LightVehicle | 3571 | PCA | -0.657 | 0.232 | 1.000 |
| B3 | LightVehicle | 3571 | PCA | -0.013 | 0.007 | 1.000 |
| B1 | Person | 2388 | PCA | -0.743 | 0.254 | 1.000 |
| B2 | Person | 2388 | PCA | -0.707 | 0.274 | 1.000 |
| B3 | Person | 2388 | PCA | -0.020 | 0.007 | 1.000 |
| B1 | All classes | 30612 | PCA | 0.827 | 0.232 | 0.871 |
| B2 | All classes | 30612 | PCA | 0.836 | 0.228 | 0.869 |
| B3 | All classes | 30612 | PCA | 0.029 | 0.010 | 0.868 |
| B5 | All classes | 30612 | UMAP n=50 | -0.734 | 0.219 | 0.868 |

Conclusion:

- Canonical fixed-size crops alone did not remove size leakage.
- Canonical fixed-size crops plus size-bias residualization reduced the strongest PCA size-axis correlation from `0.827-0.836` to `0.029` on the all-class run while preserving nearest-neighbor purity (`0.868` vs `0.869-0.871`).
- UMAP n=50 remains useful for human review, but the 2D UMAP layout itself can still show size-correlated axes. Use PCA diagnostics to decide whether the embedding recipe is size-biased.

Default recommendation:

- Encoder: DINOv3 ViT-B LVD.
- Crop: padded square, padding `0.08`.
- Preprocess: canonical `336x336`.
- DINO pooling: `pooler`.
- Embedding adjustment: `remove_size_bias`.
- Projection: PCA for bias diagnostics; UMAP n=50 for interactive review once PCA diagnostics are acceptable.

Auto-class contract:

- Trained class predictors now save the same crop/preprocess/pooling/adjustment recipe into classifier metadata.
- Auto-class inference reloads that metadata, reconstructs the same padded canonical crop, applies the saved size-bias residualizer, and then scores the classifier head.
- SAM-generated auto-class boxes now route through the same full-image bbox crop helper instead of classifying an already-cropped mask image directly.

Aggregation contract:

- `embedding_aggregation=pooled` remains the conservative default and fallback
  for Class Split and auto-class.
- `embedding_aggregation=local_salad` is valid only with a spatial-token
  encoder and an explicitly selected Tator-trained local SALAD head. DINOv3 is
  the established baseline; C-RADIOv4 heads are supported as a candidate
  encoder. The selected encoder, pooling mode, and head id are stored in Class
  Split cache keys and classifier metadata so training and inference use the
  same aggregation recipe.
- Whole-image/frame SALAD diversity scoring and object-crop SALAD aggregation
  share the same local head registry and the same no-external-checkpoint
  loader policy.

## Remaining Lever Screen

Executed on 2026-05-20 with:

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4" \
  --label-zip "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/labels/bboxes_yolo (207).zip" \
  --labelmap "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/code/labelmap.txt" \
  --image-dir "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/cropped images" \
  --output-root uploads/class_analysis/experiments/waldo_v4_remaining_smoke \
  --matrix remaining --sample-cap 50 --force
```

Artifacts:

- `uploads/class_analysis/experiments/waldo_v4_remaining_smoke/leaderboard.csv`
- `uploads/class_analysis/experiments/waldo_v4_remaining_smoke/report.md`
- `uploads/class_analysis/experiments/waldo_v4_remaining_smoke/metrics.json`

This screened 115 runs across the baseline, raw/native crops, crop geometry,
padding, background suppression, multi-view crops, DINO pooling, postprocess
transforms, UMAP settings, CLIP, selected classes, and all classes.

Key results:

- Raw/native embeddings remain rejected. On the sample screen, raw native or
  raw `224px` variants produced size-axis correlations around `0.5-0.9` on
  several single classes and around `0.77-0.81` on all classes.
- The baseline residualized recipe stayed stable: selected-class PCA size
  correlations were around `0.02-0.04`, and the sampled all-class run was
  `-0.029`.
- Padding/crop geometry variants did not produce a universal win over padded
  square `0.08`.
- Background suppression (`mean_fill`, `blur`, `darken`) did not produce a
  default-quality improvement. Keep it as an advanced diagnostic only.
- Multi-view crops preserved low size leakage but were slower. They are useful
  as a precise preset, not the default.
- DINO pooler and CLS were effectively equivalent in this run; patch mean and
  concat did not justify becoming defaults.
- Whitening is rejected as a default or normal user-facing option because it
  reintroduced bad size leakage on small classes.
- UMAP is useful for graph browsing, but its 2D axes can still look
  size-correlated. Use PCA diagnostics to evaluate the embedding recipe.

## Finalist Preset Comparison

Executed on 2026-05-20 with:

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4" \
  --label-zip "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/labels/bboxes_yolo (207).zip" \
  --labelmap "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/code/labelmap.txt" \
  --image-dir "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/cropped images" \
  --output-root uploads/class_analysis/experiments/waldo_v4_finalists \
  --matrix finalists --sample-cap 200 --force
```

Artifacts:

- `uploads/class_analysis/experiments/waldo_v4_finalists/leaderboard.csv`
- `uploads/class_analysis/experiments/waldo_v4_finalists/report.md`
- `uploads/class_analysis/experiments/waldo_v4_finalists/metrics.json`

The finalist run compared the intended UI presets on `LightVehicle`, `Person`,
`Bike`, `UPole`, `Boat`, `Truck`, `Gastank`, `Building`, and all classes.

| Preset | Recipe | Mean abs size corr | Mean runtime | Notes |
| --- | --- | ---: | ---: | --- |
| Fast | `224px`, padding `0.04`, single view, residualized | `0.032` | `15.5s` | Good quick audit and auto-class candidate. |
| Balanced | `336px`, padding `0.08`, single view, residualized | `0.031` | `17.6s` | Best default: nearly as fast as Fast with the canonical crop size. |
| Precise | `336px`, padding `0.08`, tight+context views, residualized | `0.030` | `23.3s` | Slightly better all-class purity in the sample, but slower. |

Selected checks:

| Run | Scope | Size corr | NN purity | Runtime |
| --- | --- | ---: | ---: | ---: |
| `fast_LightVehicle` | selected | `-0.027` | `1.000` | `12.8s` |
| `balanced_LightVehicle` | selected | `-0.026` | `1.000` | `12.2s` |
| `precise_tight_context_LightVehicle` | selected | `-0.072` | `1.000` | `15.8s` |
| `fast_all_classes` | all classes | `0.029` | `0.388` | `35.8s` |
| `balanced_all_classes` | all classes | `-0.031` | `0.384` | `43.8s` |
| `precise_tight_context_all_classes` | all classes | `0.019` | `0.445` | `64.8s` |

Canonical presets:

- **Fast**: DINOv3 ViT-B LVD, canonical `224x224`, padded-square crop,
  padding `0.04`, pooler readout, full crop, single view, size-bias
  residualization, PCA by default.
- **Balanced**: DINOv3 ViT-B LVD, canonical `336x336`, padded-square crop,
  padding `0.08`, pooler readout, full crop, single view, size-bias
  residualization, PCA by default. This is the conservative training default
  for auto-class and the faster Class Split audit preset.
- **Precise**: Balanced settings plus `tight_context` multi-view embeddings.
  Use when the user wants the strongest Class Split audit and can tolerate
  slower runs.

User-facing controls:

- Keep the top-level controls small: scope, class, encoder, quality preset,
  projection, and sample cap.
- Put crop padding, canonical size, DINO pooling, background mode, embedding
  views, and UMAP neighbors behind advanced controls with tooltips.
- Do not expose raw/native embeddings, whitening, or image-bias removal as
  normal presets. They remain experiment-harness options only.

## Full Uncapped Finalist Rerun

Executed on 2026-05-20 with no sample cap:

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4" \
  --label-zip "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/labels/bboxes_yolo (207).zip" \
  --labelmap "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/code/labelmap.txt" \
  --image-dir "/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/cropped images" \
  --output-root uploads/class_analysis/experiments/waldo_v4_finalists_rerun_20260520 \
  --matrix finalists --sample-cap 0 --force
```

Artifacts:

- `uploads/class_analysis/experiments/waldo_v4_finalists_rerun_20260520/leaderboard.csv`
- `uploads/class_analysis/experiments/waldo_v4_finalists_rerun_20260520/report.md`
- `uploads/class_analysis/experiments/waldo_v4_finalists_rerun_20260520/metrics.json`

All-class results:

| Preset | Projection | Object-weighted NN purity | Class-balanced NN purity | Abs size leakage | Wrong-class candidates | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Precise | PCA | `0.8969` | `0.6917` | `0.0273` | `804` | `45.3s` |
| Fast | PCA | `0.8701` | `0.6463` | `0.0308` | `1066` | `38.9s` |
| Balanced | PCA | `0.8677` | `0.6465` | `0.0292` | `1133` | `38.7s` |
| Precise | UMAP n=50 | `0.8969` | `0.6917` | `0.6332` | `804` | `54.6s` |
| Balanced | UMAP n=50 | `0.8677` | `0.6465` | `0.7367` | `1133` | `46.3s` |

A targeted full all-class rerun of the strongest sampled challenger,
`tight_standard`, did not beat the precise recipe:

| Challenger | Projection | Object-weighted NN purity | Class-balanced NN purity | Abs size leakage | Wrong-class candidates | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tight + standard | PCA | `0.8940` | `0.6858` | `0.0284` | `820` | `489.6s` |

Decision:

- Make **Precise** the Class Split Explorer default because it is the measured
  winner for all-class wrong-label discovery and class-balanced local purity.
- Keep **Balanced** as the conservative auto-class training default because it
  is single-view, faster, and easier to run repeatedly during classifier
  iteration.
- Keep **Fast** for quick coarse audits.
- Keep PCA as the diagnostic default. UMAP can still be useful for visual
  browsing, but the projection itself showed large absolute size-axis leakage
  even when the underlying embedding recipe had good PCA diagnostics.
- Use UMAP `n_neighbors=50` when UMAP is selected; this matches the finalist
  setup and is more stable than very local UMAP for this audit.
