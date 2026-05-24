# Backend Storage Hardening Log

This log keeps implementation checkpoints out of the README front page while
preserving the exact validation story for storage and artifact-write fixes.

## 2026-05-24: SAM3 and Shared Training Split Roots

- Rejected symlink components anywhere in shared Qwen/SAM3 training split roots
  before and after root creation.
- Rejected symlink components anywhere in SAM3 training run roots and direct run
  candidates.
- Rejected symlink components anywhere in SAM3 generated config roots and config
  file paths before writing generated YAML.
- Added regressions for symlinked ancestors on SAM3 split roots, SAM3 run roots,
  SAM3 generated config roots, and Qwen split roots.
- Validation: `975 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Dataset Registry and Qwen Dataset Roots

- Rejected symlink components anywhere in Qwen dataset roots and upload staging
  roots before and after root creation.
- Rejected symlink components in Qwen dataset children and upload job
  directories before finalizing uploads or creating build outputs.
- Rejected symlink components anywhere in the dataset registry root and registry
  child directories before zipped dataset copies.
- Added regressions for Qwen upload staging, Qwen finalize/build targets, and
  zipped dataset registry uploads through symlinked parent directories.
- Validation: `979 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: SAM3 Segmentation Output Root

- Added guarded SAM3 dataset root and child-directory helpers matching the Qwen
  dataset storage checks.
- Routed segmentation-build output planning through the guarded SAM3 dataset
  child helper.
- Added a regression that rejects segmentation output planning when
  `SAM3_DATASET_ROOT` has a symlinked parent.
- Validation: `980 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Qwen Training Run Roots

- Rejected symlink components anywhere in Qwen training job roots and `runs`
  roots before and after root creation.
- Rejected symlink components in Qwen run metadata result paths before writing
  `metadata.json`.
- Added regressions for Qwen run roots and metadata result paths through
  symlinked parent directories.
- Validation: `982 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Glossary Library Root

- Rejected symlink components anywhere in the glossary library root before and
  after root creation.
- Added a regression that blocks glossary saves through a symlinked root parent
  without writing to the target directory.
- Validation: `983 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: System Storage Probe Roots

- Rejected symlink components anywhere in system storage probe roots before and
  after probe root creation.
- Added a regression that blocks health/storage probes through a symlinked root
  parent without writing probe files to the target directory.
- Validation: `984 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Prompt Helper Preset Roots

- Rejected symlink components anywhere in prompt-helper preset roots before and
  after preset root creation.
- Added a regression for nested symlinked parent paths such as
  `linked_parent/nested/presets`.
- Validation: `985 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Qwen Prepass Trace Roots

- Rejected symlink components anywhere in Qwen prepass trace roots and trace
  parent directories before writing trace logs.
- Added a regression for nested symlinked parent paths such as
  `trace_parent/nested/prepass_traces`.
- Validation: `986 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Prepass Recipe Artifact Roots

- Rejected symlink components anywhere in prepass recipe roots before and after
  recipe root creation.
- Rejected symlink components in prepass recipe output-file parent paths before
  writing JSON, text, and zip artifacts.
- Added a regression for nested symlinked recipe parent paths such as
  `linked_parent/nested/recipes`.
- Validation: `987 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Agent Cascade Artifact Roots

- Rejected symlink components anywhere in agent cascade roots before and after
  cascade root creation.
- Rejected symlink components in agent cascade output-file parent paths before
  writing JSON and zip artifacts.
- Added regressions for nested symlinked cascade parent paths such as
  `linked_parent/nested/cascades`.
- Validation: `989 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: EDR Package and Runtime Stage Roots

- Rejected symlink components anywhere in EDR package roots before and after
  package root creation.
- Rejected symlink components in EDR package output-file parent paths before
  JSON, text, and zip writes.
- Rejected symlink components anywhere in runtime stage roots before staging
  package assets into classifier, detector, and calibration stores.
- Added regressions for nested symlinked package and stage parent paths.
- Validation: `991 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Output File Preflight and Calibration Roots

- Rejected symlink components in calibration job roots and calibration JSON
  output parents.
- Moved output-file symlink checks ahead of `mkdir(parents=True)` in prepass
  recipe, agent cascade, EDR package, and calibration writers so nested symlink
  ancestors cannot receive newly created directories before rejection.
- Added regressions that assert nested symlinked output parents stay untouched.
- Validation: `996 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Calibration Recipe Registry Roots

- Rejected symlink components in recipe-registry cache roots, discovery roots,
  registry roots, JSON/text output parents, and lock parents.
- Moved registry writer and lock symlink checks ahead of parent creation.
- Added regressions for nested symlinked cache parents, registry output
  parents, text output parents, and discovery locks.
- Validation: `1001 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Canonical Completion and Calibration Helper Writes

- Rejected symlink components in canonical deployment roots, canonical JSON
  output parents, canonical copy destinations, and calibration worker record
  parents.
- Moved canonical JSON/copy and calibration record checks ahead of parent
  creation; safe-link cache helpers now skip nested symlinked parents.
- Added regressions for nested symlinked canonical output parents, copy
  destinations, deployment roots, calibration record parents, and safe-link
  cache parents.
- Validation: `1006 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Dataset, Classifier, and Qwen Caption IO Roots

- Rejected symlink components in dataset metadata output parents, COCO-to-YOLO
  label roots and label output parents, classifier registry roots, and Qwen
  caption IO log parents.
- Moved dataset and caption IO parent checks ahead of parent creation.
- Added regressions for nested symlinked dataset metadata parents,
  COCO-to-YOLO label roots, classifier registry parents, and caption IO log
  parents.
- Validation: `1010 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.
- Residual direct-parent symlink scan:
  `rg -n "\.parent\.is_symlink\(|is_symlink\(\) or .*\.parent\.is_symlink\(|\.is_symlink\(\) or .*parent" services api localinferenceapi.py utils models tools tests`
  returned no hits.

## 2026-05-24: Materialized Dataset Cache Roots

- Rejected symlink components in materialized dataset allowed roots, target
  roots, and target parents before SAM3 and YOLO annotation-overlay cache
  resets.
- Added regressions for nested symlinked allowed roots and nested symlinked
  target ancestors so cache reset cannot create directories outside the allowed
  materialization root.
- Validation: `1012 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Shared Local Artifact Copy Helpers

- Rejected symlink components in shared artifact copy destination parents before
  creating parent directories or writing copied artifacts.
- `link_or_copy` now rejects nested symlinked destination parents; startup
  artifact mirroring skips them rather than writing through them during import.
- Added regressions for nested symlinked API copy, link-or-copy, and startup
  copy parents.
- Validation: `1015 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Caption, Prepass Trace, and Calibration Cache Parents

- Kept README update tracking collapsed and reduced it to the latest validation
  count plus a link to this log.
- Rejected symlink components before creating Qwen caption per-run log parents,
  Qwen prepass trace file parents, calibration job directories, and calibration
  prepass/features/labeled cache directories.
- Added regressions for nested symlinked Qwen caption log parents, Qwen prepass
  trace file parents, and calibration cache parents.
- Validation: `1018 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: EDR Package Copy Destinations

- Removed the remaining raw parent creation before EDR package asset copies so
  destination parents are preflighted by the hardened output-file helper first.
- Hardened runtime tree staging so symlink components inside a destination tree
  are skipped before parent creation and file copy.
- Added regressions for nested symlinked EDR asset-copy parents and internal
  symlinked runtime tree parents.
- Validation: `1020 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Agent Cascade Classifier Imports

- Rejected symlink components in raw cascade classifier import roots before
  creating `classifiers/imports/<tag>` directories.
- Rejected symlink components in classifier import destination parents before
  creating nested directories or writing imported classifier files.
- Added a regression for an `imports` symlink that points back inside the
  classifier store and would otherwise redirect imported files.
- Validation: `1021 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Prepass Recipe Tree Copy Destinations

- Rejected symlink components in prepass recipe filtered tree-copy destination
  roots before creating directories for exported/imported recipe assets.
- Added a regression for nested symlinked destination parents so recipe
  tree-copy cannot create directories through a linked parent before rejection.
- Validation: `1022 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Prepass Recipe Import Destinations

- Added a guarded prepass recipe directory helper for imported run bundles,
  Qwen runs, classifier upload parents, calibration job copies, and canonical
  recipe artifact directories.
- Rejected symlink components before creating those import destination
  directories.
- Added a regression for classifier imports through a nested symlinked upload
  root parent.
- Validation: `1023 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: SAM3 Training Split Parent Creation

- Routed SAM3 training split parent creation through the shared guarded split
  helper, matching the Qwen split root behavior.
- Removed the direct parent `mkdir` from SAM3 split preparation.
- Added a prepare-level regression for nested symlinked SAM3 job-root parents.
- Validation: `1024 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Detector Runtime Tree Copy Parents

- Rejected symlink components in nested detector tree-copy destination parents
  before creating directories inside staged YOLO/RF-DETR runtime trees.
- Added a regression for internal symlinked destination parents inside an
  otherwise valid detector copy root.
- Validation: `1025 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Linked Annotation Storage Roots

- Kept README update tracking collapsed and reduced to one validation sentence
  plus the detailed-log link.
- Rejected symlink components in linked dataset annotation metadata and overlay
  storage roots before creating backend-owned directories.
- Added labelmap writes to the same allowed-root and parent symlink preflight
  used by other dataset output writers.
- Added regressions for symlinked registry parents on overlay and metadata
  writes, asserting outside targets are not created.
- Validation: `1027 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Class Analysis Cache Roots

- Rejected symlink components in class-analysis cache roots before cached
  embedding loads, thumbnail-cache reuse, and cache writes.
- Routed thumbnail output directory creation through the class-analysis guarded
  directory helper instead of a raw `mkdir`.
- Added regressions for symlinked cache roots on embedding-cache reads and
  thumbnail-cache reuse, ensuring outside cache targets are not trusted.
- Validation: `1029 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Startup Storage Root Initialization

- Replaced raw import-time `mkdir` calls for backend storage roots with a shared
  initializer that rejects symlink components before and after root creation.
- Guarded the early startup artifact mirror so default CLIP artifact copies are
  skipped when `uploads/` or mirror children have symlinked components.
- Added regressions proving nested directories are not created through
  symlinked storage-root parents while normal roots still initialize.
- Validation: `1031 passed, 17 skipped`; live endpoint map/method checks and
  OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Data Ingestion Reference Profile Wording

- Kept README update tracking collapsed and trimmed the visible line to the
  latest validation count plus the detailed maintenance-log link.
- Removed product-facing "local SALAD training/head" wording from Data
  Ingestion while preserving the internal API names and backend model flow.
- Clarified the user-facing flow as reference-profile building followed by
  candidate diversity review.
- Validation: `1031 passed, 17 skipped`; focused Data Ingestion/UI contract
  checks passed.

## 2026-05-24: Data Ingestion Backend Dataset Media Roots

- Kept Data Ingestion reference/training media collected from backend datasets
  inside the selected dataset root, even when an `images/` directory is a
  symlink to another location.
- Added regression coverage for symlinked backend dataset image roots so
  reference-profile jobs cannot silently ingest files outside the dataset.
- Validation: `1033 passed, 17 skipped`; focused Data Ingestion and linked
  dataset coverage passed.

## 2026-05-24: Qwen Dataset Upload Finalization Sources

- Revalidated Qwen dataset upload jobs at finalize time against the guarded
  upload staging root instead of trusting the in-memory job root.
- Rejected symlinks inside staged upload trees before moving completed Qwen
  datasets into the training dataset store, while preserving safe replacement
  of stale top-level metadata and labelmap files.
- Added regressions for forged outside staging roots and post-upload split
  symlink swaps.
- Validation: `1035 passed, 17 skipped`; focused Qwen upload/security/backend
  coverage passed.

## 2026-05-24: Qwen Training Split Source Guards

- Required Qwen training annotations to resolve inside the selected dataset
  root before direct training or random-split materialization.
- Hardened the Qwen training image resolver so split/image directory symlinks
  cannot make direct training read images outside the dataset root.
- Added regressions for escaped annotation files and symlinked image roots.
- Validation: `1038 passed, 17 skipped`; focused Qwen training/path coverage
  passed.

## 2026-05-24: Detector Training Artifact and Split Guards

- Moved README demo placeholders and the workflow/API table into closed
  details blocks, added a first-layer quick start, and kept update tracking to
  the latest validation count plus this log link.
- Required YOLO and RF-DETR run-artifact status checks to count only regular
  files inside the run directory, so symlinked best weights/checkpoints cannot
  make a run appear complete.
- Rejected symlinked YOLO head-graft base `best.pt` files during dry-run
  preflight before dataset resolution starts.
- Rejected RF-DETR dataset split directories that resolve outside the selected
  dataset root before creating run-local split links.
- Added regressions for detector artifact symlink escapes, symlinked YOLO
  `data.yaml` labelmap fallback, YOLO head-graft base symlink escapes, and
  RF-DETR split symlink escapes.
- Validation: `1043 passed, 17 skipped`; focused detector lifecycle/head-graft
  coverage, the full pytest suite, and live endpoint sanity checks passed.

## 2026-05-24: Segmentation Build Output Guards

- Revalidated segmentation-builder output dataset roots at worker time instead
  of trusting the path string produced during planning.
- Rejected symlink swaps and pre-existing output directories before creating
  segmentation output trees.
- Stopped backfilling source `labelmap.txt` files during segmentation builds;
  the worker now copies a safe source labelmap when present or writes the class
  list directly into the guarded output dataset.
- Routed generated segmentation label files through a root-contained text writer
  so output label parents cannot be swapped through symlinks.
- Added regressions for post-plan output symlink swaps and source labelmap
  symlink escapes.
- Validation: `1045 passed, 17 skipped`; focused SAM3/segmentation coverage,
  the full pytest suite, and live endpoint sanity checks passed.

## 2026-05-24: Calibration YOLO Image Resolution

- Fixed calibration/EDR image discovery so standard YOLO split datasets under
  `train/images` and `val/images` are included instead of being reported as
  missing.
- Kept calibration cache keys stable by using relative image paths and
  deduping the same relative path across validation and training splits.
- Taught calibration prepass workers, including the single-process fallback, to
  resolve those relative paths from `split/images` before falling back to
  legacy direct split-root images.
- Rejected traversal paths, symlinked image roots, and symlinked image files
  during calibration image discovery/resolution.
- Added `.webp` to calibration image discovery to match the broader dataset
  image support used elsewhere in the backend.
- Added regressions for YOLO split-image discovery, nested image resolution,
  worker prepass reads from `val/images`, and symlink/traversal escape handling.
- Validation: `1051 passed, 17 skipped`; focused calibration/EDR coverage,
  the full pytest suite, and live endpoint sanity checks passed.

## 2026-05-24: Annotation Snapshot Record Guards

- Collapsed README update tracking into a short closed detail so the first
  layer stays focused on product overview and setup.
- Required persistent and transient annotation snapshot records to resolve to
  real dataset images before writing overlay labels or captions.
- Fixed persistent annotation snapshots so text-only records update captions
  without clearing existing bbox/segmentation overlay labels.
- Added regressions for missing-image snapshot writes and text-only persistent
  caption updates.
- Validation: `1054 passed, 17 skipped`; focused linked annotation and
  auto-label runner coverage, the full pytest suite, and live endpoint sanity
  checks passed.

## 2026-05-24: Split Dataset Caption Overlay Scoping

- Made annotation text/caption overlays split-aware for YOLO split datasets so
  `train/images/foo.jpg` and `val/images/foo.jpg` can carry distinct captions.
- Preserved legacy flat `text_labels/...` lookup as a fallback for existing
  datasets and non-split flows.
- Export now maps split-scoped caption overlays back to
  `train/text_labels/...` or `val/text_labels/...` in downloaded split dataset
  archives instead of leaking overlay storage paths.
- Added regressions for split-layout snapshot caption writes, manifest reads,
  and dataset ZIP export.
- Validation: `1056 passed, 17 skipped`; focused linked annotation,
  auto-label runner, dataset download cleanup coverage, the full pytest suite,
  and live endpoint sanity checks passed.

## 2026-05-24: Qwen Dataset Labelmap Source Guards

- Hardened YOLO-to-Qwen dataset builds so `yolo_labelmap_path` is validated
  against the source dataset root and guarded metadata root before it is read.
- Rejected symlinked labelmap files and symlinked linked-dataset registry roots
  before creating any Qwen output dataset directories.
- Kept the existing class-list fallback when no safe labelmap file exists.
- Added regressions for outside labelmap paths, symlinked labelmaps, and
  symlinked registry metadata roots.
- Validation: `1059 passed, 17 skipped`; focused linked annotation,
  Qwen dataset upload/training/runtime coverage, the full pytest suite, and
  live endpoint sanity checks passed.

## 2026-05-24: SAM3 Materialized View Metadata Root Guard

- Hardened SAM3 annotation-overlay materialization to use the guarded dataset
  metadata root before creating the `sam3_materialized` view.
- Rejected symlinked linked-dataset registry roots before any materialized
  SAM3/COCO view can be created outside the registry metadata tree.
- Added regression coverage proving a symlinked registry root is rejected and
  the outside target remains untouched.
- Validation: `1060 passed, 17 skipped`; focused linked annotation/SAM3
  lifecycle coverage, the full pytest suite, and live endpoint sanity checks
  passed.

## 2026-05-24: Linked Dataset Labelmap Source Guards

- Centralized dataset-root labelmap discovery through a root-contained,
  symlink-aware reader for `labelmap.txt`, `classes.txt`, and `labels.txt`.
- Rejected symlinked linked-dataset `labelmap.txt` files during strict
  register/open flows before registry records or transient sessions are
  created.
- Preserved ordinary in-root labelmap discovery for linked datasets and model
  materialization fallbacks.
- Added regressions for strict path registration and transient path opening
  with symlinked labelmaps.
- Validation: `1062 passed, 17 skipped`; focused linked dataset, dataset
  upload, SAM3, and Qwen coverage plus the full pytest suite and live endpoint
  sanity checks passed.

## 2026-05-24: EDR Package Runtime Source Boundaries

- Kept README update tracking collapsed and reduced the visible content to one
  concise status sentence plus this log link.
- Stopped EDR runtime resolution from repairing incomplete feature contracts by
  reading host paths embedded in imported package metadata; trusted local
  package materialization can still fill contracts before packaging.
- Required packaged YOLO/RF-DETR detector run IDs to validate as direct child
  names and resolve inside their configured run roots before any files are
  copied into an EDR package.
- Added regressions for external feature repair attempts during runtime
  resolution and detector run traversal during package materialization.
- Validation: `1063 passed, 17 skipped`; focused EDR package coverage,
  canonical EDR completion coverage, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Prepass Recipe Export Source Boundaries

- Required saved prepass/EDR export asset collection to resolve YOLO, RF-DETR,
  and calibration job directories as direct children inside their configured
  roots before copying files into portable recipe archives.
- Rejected symlinked Qwen model roots and skipped Qwen model registry entries
  whose resolved run paths live outside the Qwen job root.
- Added regressions for symlinked detector runs, symlinked calibration jobs,
  external Qwen model paths, and symlinked Qwen roots during recipe export.
- Validation: `1067 passed, 17 skipped`; focused prepass recipe export/import
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Agent Mining Cache and Auto-Label Result Roots

- Rejected symlinked agent-mining detection cache roots before sample-cache
  reads/writes, cache-size scans, or cache purges can touch files outside the
  configured cache tree.
- Replaced direct auto-label result writes with a guarded result writer that
  rejects symlinked job roots and atomically replaces stale result files.
- Added regressions for symlinked agent-mining cache sample roots, cache-size
  roots, cache-purge roots, and auto-label result roots.
- Validation: `1071 passed, 17 skipped`; focused agent-mining/auto-label
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Detector Training Cleanup Symlink Guards

- Hardened YOLO/RF-DETR failed-run cleanup so swapped symlink run directories
  are unlinked only when the symlink entry itself lives under the configured job
  root and the parent path has no symlink components.
- Added regressions proving cleanup does not delete another in-root detector run
  through a symlink target and does not unlink out-of-root symlink paths.
- Validation: `1073 passed, 17 skipped`; focused detector lifecycle/path
  containment coverage, the full pytest suite, and live endpoint sanity checks
  passed against the restarted backend.

## 2026-05-24: Dataset Registry Delete Parent Guards

- Rejected dataset delete targets whose parent path contains a symlink before
  resolving or removing linked/managed registry records.
- Preserved the existing behavior that unlinks a final registry-record symlink
  without deleting its target.
- Added linked and managed dataset regressions proving a symlinked registry
  parent cannot delete another in-root dataset tree.
- Validation: `1075 passed, 17 skipped`; focused dataset/Qwen lifecycle
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Qwen Upload Cancel Parent Guard

- Hardened Qwen dataset upload cancellation so stale upload jobs are removed
  from memory without traversing a symlinked staging parent during filesystem
  cleanup.
- Preserved safe cleanup of a final symlinked staging job entry by unlinking the
  symlink itself without touching its target.
- Added a regression proving cancellation does not delete a target reached
  through a swapped staging-parent symlink.
- Validation: `1076 passed, 17 skipped`; focused Qwen upload coverage, the full
  pytest suite, and live endpoint sanity checks passed against the restarted
  backend.

## 2026-05-24: Agent Recipe and Cascade Delete Parent Guards

- Rejected agent recipe and cascade delete targets whose nested parent path
  contains a symlink before resolving or unlinking JSON, ZIP, or recipe
  directory artifacts.
- Preserved safe cleanup of final symlinked recipe/cascade artifacts by
  unlinking the symlink entry itself.
- Added regressions proving nested symlink parents cannot delete another
  in-root recipe or cascade target.
- Validation: `1078 passed, 17 skipped`; focused recipe/cascade coverage, the
  full pytest suite, and live endpoint sanity checks passed against the
  restarted backend.

## 2026-05-24: CLIP Training Temp Cleanup Guard

- Hardened CLIP/classifier training cleanup so `job.temp_dir` cleanup only
  removes owned `clip_train_` staging directories directly under the system
  temp root.
- Prevented arbitrary `staged_temp_dir` form values from being used as broad
  filesystem cleanup targets.
- Preserved cleanup of owned final symlink entries by unlinking the symlink
  without touching its target.
- Added regressions for unowned cleanup paths, final symlink cleanup, and the
  existing class-analysis staging cleanup contract.
- Validation: `1080 passed, 17 skipped`; focused CLIP/job-start/class-analysis
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Agent Import Endpoint Staging Guards

- Hardened agent recipe and cascade import endpoints so staging directories are
  created only after the configured recipe/cascade root is proven free of
  symlink components.
- Prevented endpoint-level temp writes under symlinked import roots before the
  service import validators run.
- Closed upload handles when import staging setup fails before `_write_upload_file`
  takes ownership of the stream.
- Added endpoint regressions for symlinked recipe/cascade import roots proving
  no outside staging directory is created and the upload handle is closed.
- Validation: `1082 passed, 17 skipped`; focused agent import endpoint/service
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Dataset Glossary Metadata Guards

- Routed dataset glossary saves through the guarded metadata storage root so
  symlinked linked-dataset registry parents are rejected before any metadata
  directory or JSON file is created.
- Rejected symlinked dataset metadata files during glossary updates instead of
  treating them as missing metadata and recreating the entry.
- Switched glossary metadata persistence from raw JSON writes to the existing
  guarded atomic dataset metadata writer.
- Added regressions proving symlinked registry parents and symlinked metadata
  leaves cannot write outside the dataset registry or mutate external targets.
- Validation: `1084 passed, 17 skipped`; focused linked dataset/glossary/path
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Agent Recipe CLIP-Head Artifact Guards

- Hardened agent-recipe persistence so the per-recipe directory is prepared
  through the guarded recipe-dir helper before crops, JSON, ZIP, or CLIP-head
  artifacts are written.
- Routed imported CLIP-head override files through guarded output-file
  preparation before writing `clip_head/head.npz` or `clip_head/meta.json`.
- Hardened saved CLIP-head artifact export from trained classifier heads so
  symlinked `clip_head` directories and artifact leaves are rejected or safely
  replaced before atomic NPZ/JSON writes.
- Added regressions proving a pre-seeded recipe-dir symlink and a symlinked
  `clip_head` directory cannot receive recipe artifacts or mutate external
  targets.
- Validation: `1086 passed, 17 skipped`; focused agent recipe/import/export
  coverage, module compile checks, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Ensemble Filter Scratch Guards

- Hardened the EDR/agent ensemble-filter scratch path under
  `uploads/tmp_ensemble` so symlinked scratch roots or parents are rejected
  before request JSONL, feature NPZ, or scored JSONL files are written.
- Prepared the three expected scratch leaves explicitly and cleaned only those
  owned temp files after scoring, reducing stale scratch artifacts from
  repeated EDR applications.
- Added a regression proving a symlinked `tmp_ensemble` directory cannot
  receive ensemble-filter inputs or trigger scoring subprocesses.
- Validation: `1087 passed, 17 skipped`; focused calibration/ensemble coverage,
  module compile checks, the full pytest suite, and live endpoint sanity checks
  passed against the restarted backend.

## 2026-05-24: Prompt Helper Preset Atomic Saves

- Switched SAM3 prompt-helper preset saves from direct final-path JSON writes
  to guarded temp-file writes followed by atomic replace.
- Preserved the existing symlink/root containment checks while ensuring
  serializer or disk errors do not leave partial preset JSON at the final path.
- Added a regression that forces a mid-write JSON failure and proves no partial
  preset file or temp artifact remains.
- Validation: `1088 passed, 17 skipped`; focused backend path/job-start
  coverage, module compile checks, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Glossary Library Atomic Saves

- Switched saved glossary-library entries from direct final-path JSON writes to
  guarded temp-file writes followed by atomic replace.
- Preserved replacement of a symlinked glossary leaf without writing through to
  its target, while keeping root and parent symlink checks intact.
- Added a regression that forces a mid-write JSON failure and proves no partial
  glossary file or temp artifact remains.
- Validation: `1089 passed, 17 skipped`; focused glossary/path coverage, module
  compile checks, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Text Artifact Atomic Saves

- Switched dataset, prepass recipe, EDR package, and calibration recipe-registry
  text artifact writes from direct final-path writes to guarded temp-file writes
  followed by atomic replace.
- Preserved existing symlink/root containment checks while ensuring labelmaps,
  copied canonical recipe text, and package text artifacts cannot be left
  partially written at their final paths.
- Added regressions proving temp and final symlink leaves are replaced without
  mutating external targets for all four text-writer families.
- Validation: `1093 passed, 17 skipped`; focused dataset/prepass/EDR/registry
  coverage and the full pytest suite passed.

## 2026-05-24: Detector Text Artifact Atomic Saves

- Switched detector YAML/text artifact writes from direct final-path writes to
  guarded temp-file writes followed by atomic replace.
- Kept detector path containment and symlink replacement behavior aligned with
  existing detector JSON metadata saves.
- Added a regression proving detector text temp and final symlink leaves are
  replaced without mutating external targets.
- Validation: `1094 passed, 17 skipped`; focused detector metadata IO coverage
  and the full pytest suite passed.

## 2026-05-24: Calibration Cache and Eval Atomic Saves

- Routed calibration prepass, feature, and labeled cache metadata writes through
  the guarded atomic JSON helper instead of direct final-path writes.
- Added a guarded atomic text helper for calibration eval JSON and prepass JSONL
  outputs.
- Changed prepass JSONL output assembly so an incomplete cache raises before
  replacing an existing final artifact, preventing partial final prepass files.
- Added regressions for calibration text symlink replacement and incomplete
  prepass cache preservation.
- Validation: `1096 passed, 17 skipped`; focused calibration IO coverage and
  the full pytest suite passed.

## 2026-05-24: Agent Recipe JSON and ZIP Atomic Saves

- Routed agent recipe JSON persistence through the guarded atomic JSON helper
  instead of direct `json.dump` to the final recipe path.
- Switched the sidecar portable recipe ZIP build to a guarded temp ZIP followed
  by atomic replace.
- Added regressions proving failed JSON serialization preserves an existing
  final recipe JSON and ZIP temp/final symlink leaves do not mutate external
  targets.
- Validation: `1098 passed, 17 skipped`; focused recipe persistence coverage
  and the full pytest suite passed.

## 2026-05-24: Annotation Overlay and Labelmap Atomic Saves

- Switched linked-dataset annotation overlay text writes from direct final-path
  writes to guarded temp-file writes followed by atomic replace.
- Switched annotation labelmap saves through the same guarded temp/replace
  pattern while preserving the dataset/metadata-root containment checks.
- Added regressions proving overlay text and labelmap temp/final symlink leaves
  are replaced without mutating external targets.
- Validation: `1100 passed, 17 skipped`; focused linked annotation coverage and
  the full pytest suite passed.

## 2026-05-24: Segmentation Output Atomic Label Writes

- Switched SAM3 segmentation-builder label text writes from direct final-path
  writes to guarded temp-file writes followed by atomic replace.
- Preserved the segmentation output-root containment checks while ensuring temp
  and final symlink leaves are replaced instead of written through.
- Added a regression proving generated segmentation label writes do not mutate
  external symlink targets.
- Validation: `1101 passed, 17 skipped`; focused linked annotation/segmentation
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Qwen Upload Metadata Atomic Saves

- Added a guarded text writer for Qwen dataset-upload staging files.
- Routed upload finalization labelmap, metadata, and dataset-sidecar writes
  through temp-file writes followed by atomic replace before moving the dataset.
- Added regressions proving temp/final symlink leaves are replaced without
  mutating external targets and stale metadata symlinks are not written through.
- Validation: `1103 passed, 17 skipped`; focused Qwen upload coverage and the
  full pytest suite passed, and live endpoint sanity checks passed against the
  restarted backend.

## 2026-05-24: Qwen Training Split Atomic Saves

- Shared the guarded Qwen text writer across upload staging and training split
  materialization.
- Routed random-split `annotations.jsonl` and split metadata writes through
  temp-file writes followed by atomic replace.
- Added a regression proving Qwen training split temp/final symlink leaves are
  replaced without mutating external targets.
- Validation: `1104 passed, 17 skipped`; focused Qwen training coverage and the
  full pytest suite passed, and live endpoint sanity checks passed against the
  restarted backend.

## 2026-05-24: Qwen Dataset Conversion Atomic Saves

- Routed YOLO-to-Qwen conversion annotations, labelmap, metadata, and dataset
  sidecar writes through the guarded Qwen text writer.
- Changed conversion annotation assembly from per-image append writes to
  in-memory line collection followed by atomic split-file replacement.
- Added a regression proving Qwen dataset conversion temp/final symlink leaves
  are replaced without mutating external targets.
- Validation: `1105 passed, 17 skipped`; focused linked dataset/Qwen conversion
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Materialized Annotation View Atomic Saves

- Promoted the guarded root text writer from Qwen-only use to a shared helper.
- Routed SAM3 annotation-overlay materialization and YOLO training-cache
  materialization labelmap/label writes through temp-file writes followed by
  atomic replace.
- Added a regression proving materialized-dataset temp/final symlink leaves are
  replaced without mutating external targets.
- Validation: `1106 passed, 17 skipped`; focused linked dataset/materialization
  coverage, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Class Analysis Artifact Atomic Saves

- Routed class-analysis JSON, JSONL, thumbnail, embedding cache, and NPZ result
  artifacts through guarded temp-file writes followed by atomic replace.
- Reused the same guarded binary writer for data-ingestion embedding archives and
  local SALAD head payload saves.
- Added regressions proving class-analysis JSON, binary copy, and NPZ temp/final
  symlink leaves are replaced without mutating external targets.
- Validation: `1109 passed, 17 skipped`; focused class-analysis coverage,
  `localinferenceapi.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Class Analysis Active Workspace Upload Atomic Saves

- Changed active Class Split Explorer workspace image uploads from direct final
  writes to temp-file streaming followed by atomic replace.
- Preserved upload size/quota handling and cleanup-on-bad-manifest behavior while
  replacing temp/final symlink leaves instead of writing through them.
- Added a regression covering active workspace upload temp/final symlink leaves,
  plus existing oversize and cleanup regressions.
- Validation: `1110 passed, 17 skipped`; focused active-workspace upload coverage,
  `localinferenceapi.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Qwen Dataset Chunk Upload Atomic Saves

- Added a shared guarded binary writer and reused it for class-analysis binary
  artifacts and Qwen upload staging files.
- Changed Qwen dataset chunk images from direct final writes to temp-file writes
  followed by atomic replace.
- Changed `annotations.jsonl` chunk appends to temp-copy-plus-append followed by
  atomic replace, so image cleanup and annotation persistence stay consistent.
- Added a regression proving image and annotation temp symlink leaves are replaced
  without mutating external targets while preserving existing annotation rows.
- Validation: `1111 passed, 17 skipped`; focused Qwen upload security coverage,
  `localinferenceapi.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Prepass Recipe Binary Artifact Atomic Saves

- Added guarded binary writes for prepass recipe persistence.
- Routed generated exemplar crops, imported portable CLIP head payloads, and
  embedded recipe crop payloads through temp-file writes followed by atomic
  replace.
- Added a regression proving imported CLIP head and crop temp/final symlink leaves
  are replaced without mutating external targets.
- Validation: `1112 passed, 17 skipped`; focused prepass recipe persistence
  coverage, `services/prepass_recipes.py` compile, the full pytest suite, and
  live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Cascade Classifier Import Atomic Saves

- Added guarded binary writes for agent-cascade classifier imports.
- Changed imported classifier payload copies from direct final writes to
  temp-file writes followed by atomic replace.
- Added a regression proving a failed classifier copy leaves no partial
  imported classifier artifact behind.
- Validation: `1113 passed, 17 skipped`; focused cascade import/export coverage,
  `services/agent_cascades.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Falcon Source Patch Atomic Saves

- Added guarded text writes for Falcon source patching.
- Changed Hugging Face snapshot and official Falcon source patch writes from
  direct `write_text` calls to temp-file writes followed by atomic replace.
- Added a regression proving patching a symlinked snapshot source file replaces
  the snapshot leaf without mutating the shared cache blob.
- Validation: `1114 passed, 17 skipped`; focused Falcon coverage,
  `services/falcon_perception.py` compile, the full pytest suite, and live
  endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Detector Artifact Copy Atomic Saves

- Changed detector artifact copies from direct final-path `copy2` calls to
  temp-file copies followed by atomic replace.
- Preserved existing symlink leaf replacement and parent symlink rejection
  behavior for YOLO/RF-DETR artifact collection and run materialization.
- Added a regression proving a failed detector artifact copy leaves no partial
  final artifact or temp file behind.
- Validation: `1115 passed, 17 skipped`; focused detector metadata/lifecycle
  coverage, `services/detectors.py` compile, the full pytest suite, and live
  endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Shared Artifact Copy Atomic Saves

- Changed API, startup, canonical EDR completion, EDR package, and prepass recipe
  artifact copy helpers from direct final-path `copy2` calls to temp-file copies
  followed by atomic replace.
- Preserved existing source-is-destination no-op handling, symlink leaf
  replacement, and symlinked parent rejection/skip behavior.
- Added parameterized regressions proving copy failures leave no partial final
  artifact or temp file behind across the shared copy helpers.
- Validation: `1121 passed, 17 skipped`; focused copy-helper/prepass/EDR/
  canonical coverage, compile checks for changed modules, and the full pytest
  suite passed; live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Link Fallback Copy Atomic Saves

- Changed `_link_or_copy_file` fallback copies to create a hardlink or copied
  temp sibling first, then atomically replace the final destination.
- Preserved existing destinations when overwrite fallback copies fail.
- Changed calibration cache symlink fallback copies to use temp-file copies
  followed by atomic replace.
- Added regressions proving link fallback and calibration fallback copy failures
  leave no partial final artifact or temp file behind.
- Validation: `1124 passed, 17 skipped`; focused link/copy and calibration
  helper coverage, `localinferenceapi.py` and `services/calibration_helpers.py`
  compile checks, the full pytest suite, and live endpoint sanity checks passed
  against the restarted backend.

## 2026-05-24: Upload File Atomic Saves

- Changed `_write_upload_file` to stream uploads into a checked temp sibling and
  atomically replace the final destination only after size/quota checks pass.
- Preserved existing files and symlink targets when overwrite uploads fail size
  validation.
- Added regressions proving overwrite uploads replace symlink leaves without
  target writes and preserve existing destination files on oversize failure.
- Validation: `1126 passed, 17 skipped`; focused data-ingestion upload coverage,
  `localinferenceapi.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Transient Annotation Labelmap Atomic Saves

- Routed transient annotation labelmap updates through the existing guarded
  annotation labelmap writer.
- Replaced the direct `labelmap.txt` write with temp-file writes followed by
  atomic replace, including symlink leaf replacement and parent/root checks.
- Added a regression proving transient labelmap edits replace symlink leaves
  without mutating external targets or temp symlink targets.
- Validation: `1127 passed, 17 skipped`; focused linked annotation coverage,
  `localinferenceapi.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: COCO Annotation JSON Atomic Saves

- Routed shared COCO annotation JSON writes and COCO metadata backfills through
  a guarded atomic writer.
- Replaced direct `_annotations.coco.json` rewrites with temp-file writes
  followed by atomic replace, including symlink leaf replacement and symlinked
  parent rejection.
- Added focused regressions for final symlink replacement, stale temp symlink
  cleanup, and symlinked parent rejection.
- Validation: `1130 passed, 17 skipped`; focused COCO/dataset conversion
  coverage, `utils/coco.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.
