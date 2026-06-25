# Class Split Embedding Experiment Plan

Date: 2026-05-20

## Objective

Find the best Class Split Explorer embedding setup for a representative labeled
dataset by testing embedding and projection levers systematically. The goal is
to measure graph quality, size-bias leakage, wrong-class discovery quality, and
human review usefulness without baking any one project, class map, or image
domain into the product.

## Dataset Lock

Choose one dataset snapshot before running the experiment. The snapshot may be
the current Label Images workspace, a backend-managed dataset, or a local test
folder with labels and a label map.

Example environment:

```bash
export CLASS_SPLIT_DATASET_ROOT=/path/to/dataset
```

Inputs:

- Images: `$CLASS_SPLIT_DATASET_ROOT/images`
- Labelmap: `$CLASS_SPLIT_DATASET_ROOT/labelmap.txt`
- Labels: `$CLASS_SPLIT_DATASET_ROOT/labels.zip`

Record these fields before each benchmark:

- Image file count
- Label file count
- Non-empty label file count
- Empty label file count
- Total object count
- Missing images for label files
- Malformed label lines
- Image dimensions or dimension distribution
- Class counts and bbox size percentiles

## Current Baseline

Use the current Class Split pipeline as the baseline:

- Encoder: DINOv3
- Backbone: the selected DINOv3 ViT checkpoint
- Projection: PCA and UMAP
- Projection neighbors: `15`, `50`, and all neighbors when practical
- Crop normalization: canonical fixed-size crops
- Size-bias control: remove size/aspect bias
- Crop padding: `0.08`
- Scoring neighbors: `15`
- Sample cap: none for final runs unless the browser would become unstable

Baseline outputs must include:

- `result.json`
- `config.json`
- `embeddings.npz`
- `metadata.jsonl`
- graph report with size-axis correlation, cache stats, class counts, and
  cluster summary

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

1. Low size-axis leakage: strongest absolute size correlation should be below
   `0.35` where possible.
2. High local semantic purity: same-class nearest-neighbor purity should improve
   or remain stable.
3. Useful clusters: clusters should separate meaningful subtypes without merely
   separating object size.
4. Human usefulness: the top outliers or wrong-class candidates should contain
   enough real issues or meaningful subtype differences to justify the setup.
5. Runtime acceptable on the full selected dataset: cache use should make reruns
   practical.

## Stage 0 - Build The Experiment Harness

Add or use a backend or CLI harness that can run Class Split configurations
against a fixed dataset snapshot without using the browser.

Required behavior:

- Create a manifest from the image folder, label zip, and label map.
- Run one class at a time and all classes.
- Write each run under `uploads/class_analysis/experiments/<dataset_slug>/<run_name>/`.
- Reuse the Class Split crop and embedding cache.
- Emit `metrics.json`, `metrics.csv`, and a small `report.md`.
- Support resumable runs: skip runs with complete artifacts unless `--force`.
- Support `--dry-run` to list planned runs.

Initial command shape:

```bash
./.venv-macos/bin/python tools/run_class_split_experiments.py \
  --dataset-root "$CLASS_SPLIT_DATASET_ROOT" \
  --label-zip "$CLASS_SPLIT_DATASET_ROOT/labels.zip" \
  --labelmap "$CLASS_SPLIT_DATASET_ROOT/labelmap.txt" \
  --image-dir "$CLASS_SPLIT_DATASET_ROOT/images" \
  --output-root uploads/class_analysis/experiments/example_dataset
```

## Stage 1 - Sanity And Bias Baselines

Run these on all classes and on a balanced subset of representative classes:

- a high-count contextual class
- a high-count object class
- one or more tiny-object classes
- one or more low-count classes
- one class where hidden subclasses are expected

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

- If B3 does not materially reduce size correlation compared with B1/B2, fix
  residualization before running larger experiments.
- If UMAP n=15/50/all produces similar graph quality, keep n=50 as the default
  for review because it is usually more stable than very local UMAP.

## Stage 2 - Canonical Crop Geometry

Before pooling experiments, test canonical crop geometry directly.

Canonical crop variants:

- `canonical_224_occ95`: final size 224, object fills most of the canvas.
- `canonical_336_occ85`: fixed crop with moderate context.
- `canonical_448_occ75`: larger input with more context and less object
  occupancy.
- `canonical_336_occ65`: context-heavy crop for classes that need immediate
  surroundings.

Gate:

- Prefer the occupancy setting that lowers size-axis leakage without removing
  useful subtype/context information.
- If tiny classes need high occupancy and contextual classes need lower
  occupancy, keep occupancy as a per-run/per-class option instead of forcing one
  universal value.

## Stage 3 - DINO Pooling Modes

Implement and test DINO pooling modes:

- `pooler`: default when available
- `cls`: CLS token
- `patch_mean`: mean over patch tokens
- `cls_patch_concat`: L2-normalized concat of CLS and patch mean
- `last4_patch_mean`: average patch means from the last four hidden layers when
  available

Matrix:

- Fixed: canonical crop, remove size bias, padding `0.08`, UMAP n=50, PCA
  companion run.
- Vary: pooling mode.
- Scope: all classes plus representative single-class runs.

Gate:

- Prefer the pooling mode that reduces size-axis leakage and improves same-class
  neighbor purity.
- If a pooling mode helps small-object classes but hurts large contextual
  classes, keep pooling as a preset option rather than forcing one universal
  default.

## Stage 4 - Background And Multi-View Tests

Test whether the model is using object pixels or background shortcuts:

- no masking
- mean-fill outside bbox
- blur outside bbox
- darken outside bbox
- tight crop only
- standard crop only
- tight + standard
- standard + context

Gate:

- Reject modes that improve metrics by stripping useful context needed for human
  review.
- Prefer modes that keep neighbor purity stable while reducing background-only
  clusters and obvious size-axis separation.

## Stage 5 - Projection And Review Defaults

Compare projection modes for the task they serve:

- Global PCA: fastest baseline and easiest to compare across runs.
- Class-balanced PCA: safer all-class overview when one class dominates.
- Between-class PCA: class-separation overview.
- Within-filter PCA: per-class or filtered local structure.
- UMAP: subclass and island search when runtime is acceptable.

Recommended review behavior:

- All-class view should default to Class-balanced PCA.
- Single-class subclass search should prefer UMAP or Within-filter PCA.
- Cluster proposals should be calculated on demand, not automatically in large
  all-class views.

## Reporting

For each finalist setup, capture:

- metrics table
- graph screenshots or exported plot artifacts
- top wrong-class vignette review sample
- cluster proposal sample
- human review notes
- runtime and cache behavior
- recommendation for default, balanced, and precise presets

Keep any dataset-specific numbers, class names, and visual audit notes in local
benchmark artifacts unless they are anonymized before being added to repository
docs.
