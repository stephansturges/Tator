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
   same registered image count, the UI first saves it through the named
   `Current upload dataset name` field and `/datasets/upload_session/*` in
   bounded image batches.
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
- Active-reference upload reuse includes the requested dataset name, so changing
  that name creates or resolves a different managed dataset instead of silently
  reusing an old auto-named upload.
- If a selected reference profile already came from a backend-backed active
  Label Images upload, candidate analysis reuses that stored dataset handle
  after confirming the dataset still exists and its image count matches the open
  workspace. This keeps reloads and profile reuse on the managed dataset path
  instead of falling back to huge multipart reference uploads.
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
5. User chooses graph projection mode. The default is class-balanced PCA for
   all-class readability; global PCA preserves the previous view;
   between-class PCA spreads class centroids for label-overview work;
   within-filter PCA recomputes axes inside the selected class or class filter;
   and UMAP is available for selected-class subclass-island searches when the
   dependency is installed.
6. If the current workspace is backend-linked or a transient server-path
   dataset, UI submits a JSON `/class_analysis/jobs` request that references
   that dataset/session directly. If it is browser-only, UI packages the current
   images and raw label lines into a transient Class Split job workspace through
   `/class_analysis/jobs/active_workspace`. That workspace exists only under the
   Class Split job directory and is not registered in Dataset Management.
7. Backend embeds object crops, writes thumbnails, points, cluster summary,
   wrong-class candidates, and compact projection metadata.
8. UI renders the graph, selected crop inspector, likely-wrong panel, report,
   and bulk relabel controls.

Flow checks:

- Class Split runs against the current annotation workspace, not Data Ingestion
  candidate files.
- Dirty annotation snapshots are flushed before running in dataset-backed mode.
- Browser-only current workspaces are transient analysis inputs. They preserve
  raw YOLO label lines for images that have not been hydrated in the browser,
  but they do not become managed datasets and do not appear in Dataset
  Management.
- The active-workspace upload has its own browser `AbortController`, so the
  Class Split Cancel button remains useful while the browser is still posting
  the transient multipart snapshot and before a backend job id exists.
- Crop inspector stays at the top of the right stack.
- Crop previews fit the available inspector and support scroll zoom.
- Plot lasso supports bulk class reassignment.
- Plot class coloring renders one trace per visible class, so legend entries,
  colors, selected points, and suspicious marker outlines stay stable after
  clicks, filter changes, and tab switches.
- Switchable PCA graph coordinates are persisted separately from the public
  result payload. The public result advertises available coordinate modes; the
  browser fetches large coordinate arrays from
  `/class_analysis/jobs/{job_id}/projection/{mode}` only when needed.
- Within-filter PCA requires a selected class or class filter. If the user
  clears the filter in an all-class run, the graph explains why the mode cannot
  overlay all classes instead of rendering an empty or misleading plot.
- Wrong-class candidates can be marked correct, skipped, discarded in batches,
  shuffled, opened in Label Images, or relabeled.
- The plot can be limited to likely wrong-class points without changing the
  analysis result or right-side review panels.
- Plot points show a floating crop preview on hover so a user can inspect an
  object before selecting it.
- Subclass cluster search is an explicit selected-class workflow, not an
  always-on graph overlay. The user runs it after selected-class analysis; the
  backend reuses saved embeddings, supports UMAP-island proposals or stricter
  embedding KMeans proposals, and reports progress through
  `/class_analysis/jobs/{job_id}/cluster_search`.
- Cluster proposals show representative crops, class mix, purity, silhouette,
  and mean outlier score. Selecting a cluster selects all visible points in that
  cluster, zooms the plot to the island, flashes the medoid, and enables the
  existing bulk class-change controls.
- Likely-wrong review can be pushed to `/mobile_review.html?session=...` for a
  phone-sized one-by-one queue. Mobile review is a live relay for the current
  desktop Class Split result. It can read context crops from backend-linked,
  transient, or active-workspace Class Split sources, but class changes,
  confirmations, and skips are recorded as session actions and must be synced
  back into the currently open Label Images workspace from the desktop UI.
- Mobile review never writes directly to a backend dataset snapshot. This keeps
  mobile edits consistent with manual `See instance` fixes in Label Images:
  both paths converge through the same open annotation state and normal save
  path.

Fix from this pass:

- The long benchmark explanation starts collapsed, reducing first-screen noise.
- All-class views do not render subclass hulls or cluster proposals. This keeps
  browser-heavy clustering out of global class-overview mode and avoids
  presenting cross-class islands as within-class subclasses.
- Selected-class subclass search has its own sensitivity, max-cluster, and
  min-size controls plus progress feedback. Search jobs are tracked separately
  from the parent analysis and are cancelled/ignored safely if a newer analysis
  result replaces the graph.
- Class Split no longer creates or reuses temporary backend datasets for its
  own browser-only active-workspace analysis. The `Workspace upload name` field,
  local storage cache, and backend-dataset reuse branch were removed from Class
  Split. Data Ingestion still uses named managed uploads for reference profiles,
  because those profiles are intentionally reusable across sessions.
- Mobile review now accepts `active_workspace` Class Split jobs for preview and
  context-crop reads. This fixes the mismatch where an all-class graph created
  from the currently open browser workspace could not be pushed to mobile even
  though it represented exactly the live dataset the user wanted to review.
- Mobile review session state is bounded in memory by TTL and session count, but
  action logs are not truncated inside an active session. This avoids silently
  dropping unsynced mobile edits while still pruning abandoned sessions.
- Mobile action log entries now include stable `action_id` and monotonic
  `sequence` values. Desktop sync deduplicates by `action_id`, then applies
  `change_class`, `confirm`, `skip`, and `skip_next` actions into the open
  Label Images workspace.
- Class-change application now trusts the matched open bbox class, not the
  stale graph point class. If a user already fixed an object manually before
  syncing mobile actions, duplicate mobile changes clear the review flag without
  moving the bbox again, decrementing the wrong class count, or marking the
  image dirty unnecessarily.
- Class Split result panels and Data Ingestion result panels now have explicit
  component-level `[hidden]` CSS rules. This guards against display rules on the
  same component accidentally making hidden results visible during tab switches
  or job startup.
- The backend no longer computes per-class subclass clusters as part of every
  all-class analysis. Explicit cluster search is selected-class only, so large
  all-class plots do not trigger expensive or browser-crashing cluster overlays.
- The selected-crop inspector and likely-wrong review vignettes now render
  source-image context crops with the object box drawn over the crop. The crop
  uses up to 50 px of added context per side for large objects and black padding
  at source-image edges.
- The likely-wrong review is a full-width queue of 12 vignettes at a time.
  Users can shuffle, skip, confirm current class, jump to the source image, or
  switch to the suggested class. Confirmed, relabeled, and skipped rows drop
  from the visible queue without automatically rerunning the expensive Class
  Split job. The local annotation state is marked dirty so the normal save path
  persists corrections.
- `See instance` resolves active-image aliases before jumping back to Label
  Images, covering backend-backed, transient, chunked-upload, and browser-only
  workspace names.
- Class Split now snapshots the last completed graph before starting a new
  analysis. If upload or job start fails, the previous graph, filters,
  selection, and review panels are restored and the failure is reported without
  replacing the graph with a stale empty placeholder.
- The projection UI now explains when to use global PCA, class-balanced PCA,
  between-class PCA, within-filter PCA, and UMAP. UMAP exposes neighbor and
  minimum-distance controls for subclass-island work, while explicit subclass
  search exposes its own UMAP island or strict embedding proposal settings.
- The Label Images shortcut explainer now lists `D` as SAM point mode
  explicitly. The class carousel used by `E/R` is lighter: it renders a bounded
  nearby class window, updates Qwen/SAM3 target selects without rebuilding full
  option lists, and debounces Class Split control refreshes.

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
- Class Split cluster-assisted bulk relabeling must never operate on hidden
  filtered-out cluster members without making that display change visible first.
- Class Split cluster-search jobs must reuse parent selected-class embeddings
  and must not mutate or replace a newer graph after analysis state changes.

## Verification Scope

Current focused verification for this review:

```bash
node --check ybat-master/ybat.js
python3 -m py_compile localinferenceapi.py api/class_analysis.py
git diff --check
.venv-macos/bin/python -m pytest tests/test_labeling_panel_layout_contract.py -q
.venv-macos/bin/python -m pytest tests/test_class_analysis.py -q
```

Final focused results for the Class Split/mobile review changes:
`tests/test_labeling_panel_layout_contract.py` passed 22 tests, and
`tests/test_class_analysis.py` passed 89 tests. Interactive browser smoke is
still a separate manual/Playwright concern; during this pass the backend was not
serving on `127.0.0.1:8080`, and the in-app Browser surface was unavailable.
