# Class Split, Data Ingestion, and Dataset Management Flow Review

This review follows the user from zero state through dataset creation, reference
profile creation, candidate ingestion, class-cluster review, and dataset cleanup.
It checks each step against the visible UI, browser state, API route, backend
artifact, and data-safety invariant.

## Product Rule

Dataset records are the stable anchor. Data Ingestion should compare new media
against a chosen accepted dataset profile. Class Split should audit the labels
currently open in Label Images. Dataset Management should create, open, link,
export, delete, and restore datasets without mutating source data unexpectedly.

## Journey 1: Create A Managed Dataset And Open It For Review

1. User opens Dataset Management.
2. User uploads a YOLO/YOLO-seg ZIP or saves the current Label Images session.
3. UI posts to `/datasets/upload`.
4. Backend validates zip paths, labelmap, labels, and writes a managed dataset
   record under `uploads/datasets`.
5. Dataset card appears with type, YOLO/COCO/Qwen/caption readiness, annotation
   progress, and action buttons.
6. User can open the dataset in Label Images for annotation review.

Flow checks:

- Upload buttons are disabled while upload is active.
- Managed delete moves the dataset to trash, not immediate destruction.
- Restore returns the dataset to the library.
- Download exports a ZIP without changing the managed source directory.

Fix from this pass:

- Dataset cards now include `Use for ingestion`, which jumps directly to Data
  Ingestion and selects that dataset as the backend reference. This removes the
  previous dead step where users had to remember the dataset id, switch tabs,
  change reference source, refresh, and select the same dataset manually.

## Journey 2: Register Or Inspect A Server-Path Dataset

1. User enters an absolute server path in Dataset Management.
2. User chooses one of three actions:
   `Open transient`, `Save transient to library`, or `Register path in library`.
3. UI posts to `/datasets/open_path`, `/datasets/transient/{id}/save`, or
   `/datasets/register_path`.
4. Backend validates the root against allowed link roots and validates YOLO
   layout without copying source images.
5. Linked records store metadata and overlays only; source files remain in place.

Flow checks:

- Linked delete removes only the library record and overlays.
- Linked roots with unavailable or non-allowlisted paths are shown as unhealthy.
- The Data Ingestion handoff is disabled for unhealthy linked roots.

## Journey 3: Build Or Import A Data Ingestion Reference Profile

1. User opens Data Ingestion.
2. User chooses reference source:
   current Label Images dataset or backend dataset.
3. User builds a reference profile or uploads a profile bundle.
4. If the current Label Images workspace is not already backend-backed with the
   same registered image count, the UI first saves it through
   `/datasets/upload_session/*` in bounded image batches.
5. UI posts profile-build jobs to `/data_ingestion/salad_train_jobs` or imports
   a ZIP through `/data_ingestion/reference_profiles/import`.
6. Backend reads the selected backend dataset, trains the local profile, records
   provenance metadata, and publishes a profile only after cancellation checks
   pass.
7. UI filters profiles so only profiles matching the selected reference dataset
   are selectable.

Flow checks:

- The reference profile is an artifact, not a source dataset copy.
- Browser-only active workspaces become managed backend datasets through a
  sidecar-backed upload session before profile training; the profile job receives
  a dataset id, not thousands of multipart reference files.
- Profile exports include checksums and reference metadata.
- Imported profiles are selected only when they match the selected reference.
- Capabilities no longer expose local filesystem paths for profile files or
  backend model diagnostics.

## Journey 4: Analyze Candidate Media And Decide What To Keep

1. User selects candidate images or videos.
2. User chooses keep fraction and video sampling controls. The candidate file
   input accepts multiple images and videos in one analysis run.
3. UI posts `/data_ingestion/jobs` with candidate files, selected profile id,
   and reference source metadata.
4. Backend extracts frames when needed, using bounded worker parallelism across
   uploaded media, then embeds candidates and reference media in batches with
   parallel image loading inside each batch. Model inference remains batched on
   the selected encoder backend.
5. Backend pools every uploaded image and every sampled video frame into one
   candidate matrix, computes the keep fraction once across that whole upload
   batch, then saves result metadata, embeddings, and thumbnails under the
   ingestion job directory.
6. UI shows report, ranked candidate list, accepted-output controls, and an
   optional reference distribution map.

Flow checks:

- Raw reference distance is diagnostic. Selection is based on farthest-first
  coverage: each candidate is initialized by distance to the reference profile,
  then selected candidates reduce the score of nearby remaining candidates so
  the kept set covers candidate-candidate diversity too. Result items expose
  `coverage_score` for selection and `reference_novelty_score` for the
  nearest-reference diagnostic.
- Local Vendi is a Vendi-style effective-rank score over each candidate
  image/frame's patch embeddings. It is computed from the same frozen
  DINOv3/C-RADIO spatial tokens used by the selected reference profile, then
  normalized within the full candidate pool. By default it contributes a light
  ranking bonus (`local_vendi_weight=0.2`) so visually dense frames can win
  close calls without replacing reference novelty or upload-batch coverage.
- Keep fraction is global to the current analysis run. `0.2` on 30 uploaded
  videos means top 20% of all sampled frames after all files are blended, not
  20% of each video.
- The UI list is ordered by `selection_priority_rank`. Kept rows are the greedy
  selection order up to the keep cutoff; rejected rows continue below that
  cutoff ordered by final `selection_score` after the kept coverage set is
  chosen. Result summaries expose `selection_score_kind` and
  `selection_score_description` so the UI can label whether that score is raw
  coverage distance or the coverage-percentile-plus-Local-Vendi priority score.
- Reference profile training uses the `strong_photometric_spatial_v2`
  augmentation profile: random resized crops, aspect/rotation/perspective
  jitter, brightness/contrast/saturation/hue/gamma changes, optional grayscale,
  blur, noise, JPEG compression, and random erasing. The UI default is 8 epochs
  so the local head sees enough hard positive views to learn those invariances.
- Candidate source paths are stripped from public result payloads.
- Reference hover thumbnails are generated lazily. Uploaded references are read
  from job media, while backend-dataset references are revalidated against the
  selected dataset root before thumbnailing. Large reference datasets do not pay
  thumbnail cost unless the user previews those points.
- Accepted output preview/download reads from the job directory only and creates
  new thumbnails or ZIPs. It does not overwrite candidate inputs.

Fixes from this pass:

- The map/details preview now has a `Keep candidate` or `Discard candidate`
  button. This updates the same accepted-item state as the checkbox in the
  candidate list, then refreshes output summary and map colors.
- The candidate list now exposes a `Show all candidates` control when results
  are capped for rendering, making the cap explicit instead of silently hiding
  candidates.
- The distribution-map panel explicitly states that Class Split Dataset Analysis
  is not required for the map.
- Hover previews on the ingestion map hide cleanly when the next point has no
  preview image, avoiding stale candidate/reference previews.

## Journey 5: Export Accepted Ingestion Output

1. User reviews accepted candidates.
2. User chooses output mode: original, resize fit, resize stretch, center crop,
   or tile.
3. User previews output geometry.
4. UI posts `/data_ingestion/jobs/{job_id}/accepted_export/preview`.
5. User downloads the accepted ZIP through
   `/data_ingestion/jobs/{job_id}/accepted_export/download`.

Flow checks:

- Explicit empty selections remain empty and do not fall back to default keep
  rows.
- Target size and total output count are bounded.
- `drop_partials` produces no output for sources smaller than the tile.
- The ZIP is intentionally unsplit so related frames or tiles can be split
  downstream by original source without train/val leakage.

## Journey 6: Run Class Split From The Current Annotation Dataset

1. User opens or loads images and labels in Label Images.
2. User opens Class Split Explorer.
3. UI reports whether images, labelmap, and labeled objects are available.
4. User chooses selected-class or all-class scope.
5. UI packages the active workspace through
   `/class_analysis/jobs/active_workspace`.
6. Backend embeds object crops, writes thumbnails, points, cluster summary, and
   wrong-class candidates.
7. UI renders the graph, selected crop inspector, likely-wrong panel, report,
   and bulk relabel controls.

Flow checks:

- Class Split runs against the current annotation workspace, not Data Ingestion
  candidate files.
- Dirty annotation snapshots are flushed before running in dataset-backed mode.
- Crop inspector stays at the top of the right stack.
- Crop previews fit the available inspector and support scroll zoom.
- Plot lasso supports bulk class reassignment.
- Wrong-class candidates can be marked correct or relabeled.

Fix from this pass:

- The long benchmark explanation starts collapsed, reducing first-screen noise.

## Journey 7: Optional Class Split Dataset Analysis

1. User runs Class Split with `Scope = All classes`.
2. User opens the Dataset Analysis panel at the bottom of Class Split.
3. User runs image-value analysis.
4. UI ranks source images by class rarity, feature rarity, and graph-edge value.

Flow checks:

- Dataset Analysis is separate from Data Ingestion distribution maps.
- It needs an all-class Class Split result, because selected-class analysis does
  not contain the global class graph needed for image-level value.
- Data Ingestion offers an `Open Class Split setup` button that switches to
  Class Split, selects all-class scope, opens Dataset Analysis for context,
  scrolls to the Class Split run setup, and explains the needed run.

Fix from this pass:

- Dataset Analysis now starts collapsed as a bottom panel instead of occupying
  the Class Split right-hand review stack or rendering large empty boxes during
  the normal cluster-audit workflow.
- Dataset Analysis graph points now show the same crop hover preview used by the
  ranked image list, so image-value outliers can be inspected before clicking.

## Backend Invariants Reviewed

- Dataset mutations must either write durable metadata or roll back.
- Linked dataset deletes must never delete linked source images.
- Managed dataset deletes must be restorable.
- Public Data Ingestion result and capability payloads must not expose internal
  source paths.
- Data Ingestion cancellation must not publish result/profile artifacts after a
  terminal cancel boundary.
- Accepted exports must be read-only with respect to submitted candidates.
- Class Split relabeling must mark the corresponding annotation image dirty so
  the normal save path persists the change.

## Verification Scope

Current focused verification for this review:

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python -m py_compile localinferenceapi.py services/data_ingestion.py
node --check ybat-master/ybat.js
git diff --check
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python -m pytest tests/test_data_ingestion.py -q
NO_ALBUMENTATIONS_UPDATE=1 .venv-macos/bin/python -m pytest tests/test_labeling_panel_layout_contract.py tests/ui/e2e/test_dataset_ingestion_safety_flows.py -q
curl -fsS http://127.0.0.1:8000/data_ingestion/capabilities
```

Interactive browser smoke is still a separate manual/Playwright concern; the
flow contract above is enforced through code contracts and targeted backend
tests.
