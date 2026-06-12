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

## 2026-05-24: SAM3 Promotion Marker Atomic Saves

- Routed the SAM3 `.promoted` marker through the guarded atomic text writer.
- Replaced the direct marker write with temp-file writes followed by atomic
  replace, including symlink leaf replacement and stale temp symlink cleanup.
- Added a regression proving SAM3 promotion markers replace symlink leaves
  without mutating external marker targets or temp symlink targets.
- Validation: `1131 passed, 17 skipped`; focused SAM3 lifecycle coverage,
  `localinferenceapi.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Prompt Helper Preset Atomic Saves

- Tightened prompt helper preset saves to replace stale symlink leaves instead
  of rejecting or following them.
- Switched temp preset creation to exclusive no-follow opens before atomic
  replace.
- Added a regression proving preset saves replace final and temp symlink leaves
  without mutating external targets.
- Validation: `1132 passed, 17 skipped`; focused prompt helper/path containment
  coverage, `services/prompt_helper_presets.py` compile, the full pytest suite,
  and live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Shared Helper No-Follow Temp Writes

- Switched shared dataset, detector, EDR package, calibration, canonical EDR,
  cascade, recipe-registry, and COCO JSON/text helpers to no-follow temp-file
  writes before atomic replace.
- Preserved existing symlink leaf replacement behavior while closing the
  check-then-open gap for temporary write targets.
- Added cleanup for additional temp write failure paths that previously relied
  on the happy path reaching `os.replace`.
- Validation: `1132 passed, 17 skipped`; focused storage/path helper coverage,
  module compile checks for the touched helpers, the full pytest suite, and
  live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Local API No-Follow Temp Writes

- Routed local API temp writes for clip-head artifacts, ensemble-filter
  JSONL inputs, auto-label results, annotation labelmaps/overlays, glossary
  entries, agent-mining sample caches, segmentation outputs, and active uploads
  through no-follow temp-file opens before atomic replace.
- Preserved existing serializer failure coverage for glossary saves while
  keeping the temp write no-follow.
- Validation: `1132 passed, 17 skipped`; focused linked annotation, glossary,
  prepass recipe, data-ingestion, calibration, and auto-label coverage,
  `localinferenceapi.py` compile, the full pytest suite, and live endpoint
  sanity checks passed against the restarted backend.

## 2026-05-24: Cross-Platform Archive Member Guards

- Rejected Windows absolute, UNC, and drive-style paths in dataset ZIP uploads,
  EDR package imports, and agent cascade imports.
- Kept existing POSIX traversal and symlink-member protections, but removed the
  host-OS assumption from archive member validation.
- Added regressions for Windows-style absolute members across dataset ZIP, EDR
  package, and agent cascade import paths.
- Validation: `1138 passed, 17 skipped`; focused dataset ZIP, EDR package, and
  agent cascade import coverage, module compile checks for touched importers,
  the full pytest suite, and live endpoint sanity checks passed against the
  restarted backend.

## 2026-05-24: Agent Cascade Import Total Size Cap

- Added a cumulative uncompressed-size cap to agent cascade archive imports.
- Reused the existing cascade archive byte limit as the default total
  uncompressed ceiling, while preserving per-entry and nested-recipe limits.
- Added a regression where individually small cascade entries exceed the
  allowed archive total and are rejected before classifier or recipe materialization.
- Validation: `1139 passed, 17 skipped`; focused agent cascade import/endpoint
  coverage, module compile checks for the cascade importer and backend wiring,
  the full pytest suite, and live endpoint sanity checks passed against the
  restarted backend.

## 2026-05-24: SAM3 Activation Path Guard

- Restricted SAM3 model activation to the configured base checkpoint or regular
  checkpoint files inside SAM3 run `checkpoints/` directories.
- Rejected arbitrary existing checkpoint-looking files outside SAM3 storage,
  checkpoint symlink escapes, invalid checkpoint suffixes, and stale active
  custom checkpoints that no longer pass the activation guard.
- Kept configured base checkpoints usable, including paths outside the SAM3 run
  root, and made base-active detection compare resolved paths.
- Collapsed the README maintainer checkpoint to one closed line so first-layer
  README content stays focused on the product and getting started.
- Validation: `1144 passed, 17 skipped`; focused SAM3 lifecycle coverage,
  `localinferenceapi.py` compile, `git diff --check`, the full pytest suite, and
  live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Detector Active Model Root Guard

- Tightened persisted YOLO and RF-DETR active-model loading so `best_path` must
  resolve inside the matching detector job root before inference can use it.
- Required active labelmap files to resolve inside the same run directory as
  the active detector weights.
- Applied the guarded YOLO active loader to inference, calibration, active-model
  listing, and active-model retrieval paths.
- Prevented RF-DETR legacy fallback active files from being migrated when their
  checkpoint paths point outside the RF-DETR run root.
- Validation: `1149 passed, 17 skipped`; focused detector metadata/active
  lifecycle coverage, `services/detectors.py` and `localinferenceapi.py` compile,
  `git diff --check`, the full pytest suite, and live endpoint sanity checks
  passed against the restarted backend.

## 2026-05-24: Qwen Runtime Settings Platform Validation

- Stopped `/qwen/settings` updates from silently accepting unknown
  `inference_platform` strings and normalizing them to `auto`.
- Preserved the supported aliases (`torch`, `transformers`, `mlx`, `mlx-vlm`,
  and `mlx_vlm`) while returning `qwen_inference_platform_invalid` for typos or
  device names such as `cuda`.
- Added focused regressions for invalid platform rejection and alias acceptance.
- Validation: `1151 passed, 17 skipped`; focused Qwen MLX/runtime settings
  coverage, `localinferenceapi.py` compile, `git diff --check`, the full pytest
  suite, and live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: SAM Preload Slot Supersession Scope

- Scoped SAM preload request-id supersession by `(slot, variant)` so concurrent
  `current`, `next`, and `previous` preloads no longer cancel each other when
  they use the same SAM variant.
- Kept generation-based supersession variant-wide so stale current-image
  navigation preloads still get skipped.
- Renamed the README's closed maintenance summary to `Update Tracking` and kept
  the first-layer content to one validation sentence plus this log link.
- Validation: `1154 passed, 17 skipped`; focused SAM preload slot coverage,
  `localinferenceapi.py` compile, `git diff --check`, the full pytest suite,
  and live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: SAM Preload Strict Slot Validation

- Stopped strict SAM slot resolution from falling back to `current` when an
  unknown slot name is submitted.
- Kept permissive fallback behavior for older internal callers that explicitly
  allow disabled or unknown slot fallback.
- Returned `slot_invalid` from preload-worker validation when a stale queued
  job somehow reaches the worker with an unknown slot.
- Validation: `1155 passed, 17 skipped`; focused SAM preload slot coverage,
  `localinferenceapi.py` compile, `git diff --check`, the full pytest suite,
  and live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Predictor Settings GPU Field UI Contract

- Fixed Qwen and SAM3 training GPU-refresh code to read the snake_case
  `/predictor_settings` fields emitted by the backend.
- Kept a camelCase fallback for any older injected payloads while making the
  backend contract explicit in UI coverage.
- Added a UI contract regression so training GPU availability cannot silently
  disappear because of field-name drift.
- Validation: `1156 passed, 17 skipped`; predictor-settings UI contract
  coverage, `node --check ybat-master/ybat.js`, `git diff --check`, the full
  pytest suite, and live endpoint sanity checks passed.

## 2026-05-24: SAM Predictor Budget Unload

- Changed SAM predictor capacity reduction to unload disabled slots instead of
  only clearing their current image state.
- Preserved token and image-index cleanup for disabled slots before unloading.
- Added a predictor-manager regression so reducing the budget verifies both
  disabled-slot unload calls and stale index removal.
- Validation: `1157 passed, 17 skipped`; focused predictor-manager and SAM
  preload coverage, `localinferenceapi.py` compile, `git diff --check`, the
  full pytest suite, and live endpoint sanity checks passed against the
  restarted backend.

## 2026-05-24: Qwen Runtime Unload Error Reset

- Cleared `qwen_last_error` during Qwen runtime unload so `/qwen/status` does
  not keep reporting stale failures after the user explicitly unloads the model.
- Propagated that reset through the top-level backend unload wrapper alongside
  model, processor, device, and caption-cache state.
- Added focused coverage for unload clearing the model state, stale error, and
  caption LRU cache.
- Validation: `1158 passed, 17 skipped`; focused Qwen runtime coverage,
  `localinferenceapi.py` and `services/qwen_runtime.py` compile,
  `git diff --check`, the full pytest suite, and live endpoint sanity checks
  passed against the restarted backend.

## 2026-05-24: Data Ingestion Profile Provenance

- Returned local SALAD head metadata from reference validation so analysis jobs
  can preserve the selected reference profile's actual base encoder.
- Fixed completed diversity-analysis summaries to report the SALAD profile's
  DINOv3/C-RADIOv4 encoder, checkpoint, C-RADIO pooling mode, and head backend
  instead of stale request defaults.
- Added a regression using a C-RADIO local SALAD profile to ensure result
  summaries ignore mismatched request-side encoder defaults.
- Validation: `1159 passed, 17 skipped`; focused Data Ingestion coverage,
  `git diff --check`, the full pytest suite, and live endpoint sanity checks
  passed against the restarted backend.

## 2026-05-24: Data Ingestion Active Profile Matching

- Tightened active Label Images reference-profile matching so profiles with
  different non-empty reference labels no longer match solely because both came
  from the active workspace flow.
- Mirrored the same label-aware filtering in the UI profile picker while keeping
  older unlabeled active-workspace profiles loadable.
- Extended backend and UI contract coverage for the active reference-label
  comparison.
- Validation: `1159 passed, 17 skipped`; focused Data Ingestion/UI contract
  coverage, `node --check ybat-master/ybat.js`, `git diff --check`, the full
  pytest suite, and live endpoint sanity checks passed against the restarted
  backend.

## 2026-05-24: Split Dataset Text Label Lookup

- Added split-aware normalization for dataset text-label endpoints so
  `train/foo.jpg` and `val/foo.jpg` resolve to split-layout source captions
  under `train/text_labels/` and `val/text_labels/`.
- Reused the same resolver for single-caption reads, batch-caption reads, and
  direct caption writes while preserving flat-dataset nested paths.
- Added linked-dataset coverage for split-prefixed source captions and kept the
  split overlay snapshot tests green.
- Validation: `1160 passed, 17 skipped`; focused linked-dataset annotation
  coverage, `localinferenceapi.py` compile, `git diff --check`, the full pytest
  suite, and live endpoint sanity checks passed against the restarted backend.

## 2026-05-24: Split Dataset Text Label Routes

- Changed dataset text-label single read/write routes to use FastAPI's path
  converter so encoded split-prefixed image names such as `train%2Ffoo.jpg`
  reach the caption resolver instead of 404ing at the router.
- Added a TestClient regression for encoded split-caption reads and writes
  through the real app route.
- Kept the README's top-level update tracking collapsed and shortened the
  visible Qwen runtime prose into a closed details section.
- Validation: `1161 passed, 17 skipped`; focused linked-dataset route coverage,
  route uniqueness coverage, `node --check ybat-master/ybat.js`,
  `api/datasets.py`/`localinferenceapi.py` compile, `git diff --check`, and the
  full pytest suite passed; live endpoint map/method checks and OpenAPI sanity
  checks passed against the restarted backend.

## 2026-05-24: Caption Dataset Ownership in Annotation Mode

- Locked the Qwen caption dataset picker to the active annotation dataset while
  Label Images is open on a dataset so manual picker changes cannot reset the
  annotation session's `textLabels` store.
- Made the effective caption dataset resolver return the active linked
  annotation dataset in annotation mode and no external dataset for transient
  sessions.
- Kept dataset refreshes from re-enabling the picker or clearing captions while
  annotation mode owns caption persistence.
- Added UI contract coverage for the locked picker, annotation-owned dataset
  resolver, and refresh path.
- Validation: `1162 passed, 17 skipped`; focused caption UI contract and
  dataset annotation coverage, `node --check ybat-master/ybat.js`,
  `git diff --check`, the full pytest suite, a Playwright annotation-mode
  smoke check, and live endpoint map/method/OpenAPI sanity checks passed
  against the restarted backend.

## 2026-05-24: Qwen Caption Window Geometry

- Made Qwen caption window sizing clamp to the decoded image dimensions so tiny
  images do not produce padded crop windows.
- Normalized caption label hints from the caller's source coordinate frame into
  the decoded image frame before full-image prompts and window grouping run.
  This keeps window crops and bbox priors aligned when the backend decodes or
  downsizes an oversized base64 image.
- Added regressions for decoded-image hint scaling and tiny-image window
  planning.
- Validation: `1164 passed, 17 skipped`; focused Qwen caption prompt/progress
  and runtime coverage, `py_compile` for Qwen/backend schema files,
  `node --check ybat-master/ybat.js`, `git diff --check`, the full pytest
  suite, and live endpoint map/method/OpenAPI sanity checks passed against the
  restarted backend.

## 2026-05-24: EDR Recipe Caption Replay

- Fixed Qwen/EDR recipe replay so loaded recipe configs that explicitly set
  `prepass_caption: false` keep captioning disabled instead of the browser
  forcing captions back on for every non-package recipe.
- Preserved the same flag when a loaded recipe is re-saved, so imported or
  canonical no-caption recipes are not silently rewritten with different
  runtime behavior.
- Added a UI contract regression covering both the live payload builder and the
  saved recipe config builder.
- Validation: `1165 passed, 17 skipped`; focused labeling panel, prepass recipe,
  and EDR runtime coverage, `node --check ybat-master/ybat.js`,
  `git diff --check`, the full pytest suite, and live endpoint
  map/method/OpenAPI sanity checks passed.

## 2026-05-24: Auto-Label Falcon Runtime Gate

- Made the auto-label runner require Falcon runtime availability only when
  `enable_falcon` is true.
- This lets baseline-only and EDR-only auto-label jobs run on machines where
  Falcon is unavailable, instead of failing before dataset or baseline work
  starts.
- Added a regression proving a no-Falcon auto-label request completes even when
  the Falcon runtime reports unavailable, and that Falcon candidate generation
  is not invoked.
- Validation: `1166 passed, 17 skipped`; focused auto-label runner, backend job
  start, and EDR runtime coverage, `py_compile localinferenceapi.py`,
  `git diff --check`, the full pytest suite, and live endpoint
  map/method/OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: YOLO CUDA Device Preflight

- Tightened YOLO training device resolution so explicit CUDA device IDs are
  rejected before queueing when CUDA is unavailable, duplicated, or outside the
  detected device range.
- This prevents Mac `auto` runs with stale CUDA IDs from creating run
  directories and failing later inside Ultralytics instead of at request
  validation.
- Added resolver coverage and backend job-start coverage proving invalid CUDA
  IDs fail before the worker or run directory is created.
- Validation: `1170 passed, 17 skipped`; focused macOS acceleration and backend
  job-start coverage, `py_compile services/detectors.py localinferenceapi.py`,
  `git diff --check`, the full pytest suite, and live endpoint
  map/method/OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: RF-DETR CUDA Device Preflight

- Added RF-DETR-specific device resolution so explicit CUDA device IDs are
  rejected before queueing when CUDA is unavailable, duplicated, or outside the
  detected device range.
- Stored the resolved RF-DETR training device in the job config and reused it in
  the worker, avoiding stale worker-only validation and Qwen-labeled CUDA error
  details.
- Added resolver coverage and backend job-start coverage proving invalid CUDA
  IDs fail before the worker or run directory is created.
- Validation: `1175 passed, 17 skipped`; focused macOS acceleration and backend
  job-start coverage, `py_compile services/detectors.py localinferenceapi.py`,
  `git diff --check`, the full pytest suite, and live endpoint
  map/method/OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Qwen Training Device Handling

- Added backend validation for Qwen Transformers training CUDA device strings,
  rejecting unavailable, duplicate, invalid, or out-of-range CUDA IDs during
  config construction instead of letting `CUDA_VISIBLE_DEVICES` drift into the
  worker.
- Cleared CUDA-only device selections for MLX Qwen training and made the
  training runner leave `CUDA_VISIBLE_DEVICES` untouched for MLX runs.
- Added config-level coverage for CUDA device normalization/rejection and
  MLX-device clearing, plus runner coverage proving MLX training does not
  mutate the CUDA environment.
- Validation: `1180 passed, 17 skipped`; focused Qwen runtime/training/backend
  job-start coverage, `py_compile localinferenceapi.py tools/qwen_training.py`,
  `git diff --check`, the full pytest suite, and live endpoint
  map/method/OpenAPI sanity checks passed against the restarted backend.

## 2026-05-24: Linked Dataset Labelmap Overlays

- Routed linked-dataset labelmap saves to the registry record instead of the
  source dataset root, so class-list edits cannot overwrite a user's
  `labelmap.txt` in a linked source tree.
- Kept transient server-path labelmap edits in the transient session until the
  user explicitly saves the session to the library; saving then writes the
  labelmap snapshot under the registry record, not the source folder.
- Made dataset listing prefer registry-owned labelmap overrides and return the
  effective class list, so SAM3/YOLO materialization sees the saved override
  without mutating source files.
- Validation: `1182 passed, 17 skipped`; focused dataset annotation, linked-root,
  zip upload, data ingestion, dataset metadata, Qwen dataset upload, COCO/YOLO,
  Qwen runtime/training, and backend job-start coverage passed, plus
  `py_compile` for the changed backend modules.

## 2026-05-24: Managed Dataset Trash/Restore

- Changed managed dataset deletion from immediate tree removal to a backend
  trash move under the owning dataset root, with symlink and root-containment
  checks on trash creation, listing, and restore.
- Added `GET /datasets/trash` and `POST /datasets/trash/{trash_id}/restore`,
  including unique-id restore behavior when the original dataset id has already
  been reused.
- Added Dataset Management UI controls to list deleted managed datasets and
  restore them, while keeping linked dataset deletion as registry/overlay-only.
- Added rollback coverage for failed restore metadata writes so a dataset moved
  out of trash is moved back instead of becoming half-restored.
- Validation: `1187 passed, 17 skipped`; focused dataset/data-ingestion safety
  bundle (`189 passed`), `py_compile localinferenceapi.py api/datasets.py`,
  `node --check ybat-master/ybat.js`, `git diff --check`, and the full pytest
  suite passed. Restarted backend on `127.0.0.1:8000` and live-checked
  `/datasets`, `/datasets/trash`, `/data_ingestion/capabilities`, OpenAPI route
  methods for the trash endpoints, and UI endpoint method matching.

## 2026-05-24: Qwen Dataset Delete Joins Trash/Restore

- Routed `DELETE /qwen/datasets/{dataset_id}` through the same managed
  dataset trash/restore path used by Dataset Management deletes.
- Preserved the active-training guard and symlink rejection before any move, so
  Qwen datasets referenced by running jobs stay protected and symlinked dataset
  entries cannot redirect deletion.
- Added regression coverage proving Qwen dataset delete moves the dataset into
  managed trash and restores it with `metadata.json` updated.
- Validation: `1188 passed, 17 skipped`; focused Qwen lifecycle and dataset
  trash/route/UI contract coverage (`100 passed`), `py_compile
  localinferenceapi.py api/qwen_datasets.py api/datasets.py`,
  `node --check ybat-master/ybat.js`, `git diff --check`, and the full pytest
  suite passed. Restarted backend on `127.0.0.1:8000` and live-checked
  `/qwen/datasets`, `/datasets/trash`, OpenAPI route methods for Qwen/trash
  deletes, and UI endpoint method matching.

## 2026-05-24: UI Endpoint Checker Dynamic Detector Routes

- Fixed `tools/check_ui_endpoints.py` to expand the UI's detector `endpoint`
  template variable into both `yolo` and `rfdetr` before comparing fetch calls
  against OpenAPI.
- This removes false missing-path reports for shared detector run list,
  summary, active, download, and delete calls while keeping method checks strict.
- Added unit coverage proving dynamic detector run paths expand to concrete
  OpenAPI-style routes instead of the generic `/{}/runs` form.
- Validation: `1189 passed, 17 skipped`; focused endpoint-checker, route
  uniqueness, and UI contract coverage (`3 passed`), `py_compile
  tools/check_ui_endpoints.py`, `git diff --check`, live
  `tools/check_ui_endpoints.py http://127.0.0.1:8000` with no missing paths or
  method mismatches, and the full pytest suite passed.

## 2026-05-24: Validation Cleanup Handles Dataset Trash

- Updated the GPU validation cleanup runner to consume `trash_path` from
  generated dataset deletes and remove only run-scoped trash entries.
- Updated the UI contract runner to parse dataset delete JSON and purge only
  its own `ui_contract_upload_*` trash entries under local `uploads/`.
- Added tool-level regressions proving generated validation trash is removed
  while non-contract or non-run-scoped trash is preserved.
- Validation: `1193 passed, 17 skipped`; focused cleanup/tooling coverage
  (`6 passed`), `py_compile tools/run_gpu_validation_suite.py
  tools/run_ui_contract_tests.py tools/check_ui_endpoints.py`,
  `git diff --check`, live UI endpoint check with no missing paths or method
  mismatches, and the full pytest suite passed.

## 2026-05-24: Validation Cleanup Path Containment

- Replaced string-prefix cleanup allow-list checks in the GPU validation runner
  with path-aware containment checks, so sibling paths such as `uploads_evil`
  or `tmp_evil` cannot be removed just because their names share a prefix with
  an allowed root.
- Added a regression proving `_safe_remove_path` rejects an
  `uploads_evil/unit_payload` sibling even when it contains the current run id.
- Validation: `1194 passed, 17 skipped`; focused dataset/data-ingestion safety
  coverage (`184 passed`), cleanup-tool coverage (`5 passed`), live UI endpoint
  map check with no missing paths or method mismatches, live OpenAPI sanity
  (`tested=144`, `failures=[]`), `py_compile tools/run_gpu_validation_suite.py`,
  and the full pytest suite passed.

## 2026-05-24: Dataset and Data Ingestion UI Safety Coverage

- Made the annotation close-block E2E deterministic by forcing all snapshot
  saves to fail until the close guard is observed, covering blur/autosave and
  close-time saves.
- Added browser coverage for managed Dataset Management delete/restore:
  a uniquely-prefixed test dataset is uploaded, deleted into backend trash,
  restored through the UI, and then purged from test-owned storage.
- Added browser coverage for Data Ingestion reference/profile gating:
  backend reference mode enables only with a selected dataset, analysis stays
  disabled until a matching saved reference profile and candidate file are
  present, C-RADIOv4 model selection is enabled only for the C-RADIO training
  encoder, and backend-reference analysis posts no browser reference files.
- Added Data Ingestion to the all-tabs navigation E2E contract and a multipart
  API helper for UI tests that need realistic dataset uploads.
- Validation: targeted UI E2E (`10 passed`), focused dataset/data-ingestion
  safety bundle (`158 passed`), `py_compile` for the changed UI test helpers,
  and the full pytest suite (`1194 passed, 19 skipped`) passed against the
  running backend.

## 2026-05-24: Prepass Recipe Delete Symlink Parents

- Hardened prepass recipe path resolution so delete/export/load helpers reject
  any symlink component in the resolved recipe directory path, not just a
  symlink as the recipe leaf.
- Added regressions proving a symlinked recipe directory and a symlinked nested
  parent cannot make prepass recipe delete remove the symlink target.
- Validation: focused prepass delete regressions (`3 passed`), broader
  destructive-endpoint safety bundle (`170 passed`), `py_compile
  services/prepass_recipes.py tests/test_prepass_recipe_config_validation.py`,
  `git diff --check`, and the full pytest suite (`1196 passed, 19 skipped`)
  passed.

## 2026-05-24: Dataset Delete Active-Job Guards

- Added dataset-id based active-job blocking before dataset delete, so linked
  dataset records and overlays cannot be removed while a running backend job
  still references that dataset id.
- Narrowed path-root delete blocking to actually active job states
  (`queued`, `running`, `cancelling`) and treated `completed` as terminal, so
  historical completed jobs do not permanently block safe managed trash moves.
- Added regressions for linked dataset delete blocked by an active Data
  Ingestion job and managed dataset delete allowed after a completed job
  reference.
- Validation: focused dataset/data-ingestion/storage safety bundle
  (`149 passed`), live browser dataset/ingestion E2E (`9 passed`),
  `py_compile localinferenceapi.py tests/test_dataset_linked_annotation_flows.py
  tests/test_data_ingestion.py`, `node --check ybat-master/ybat.js`,
  `git diff --check`, live UI endpoint map check with no missing paths or
  method mismatches, live OpenAPI sanity (`tested=144`, `failures=[]`), and the
  full pytest suite (`1198 passed, 19 skipped`) passed.

## 2026-05-24: Dataset Metadata Write Rollback

- Added a focused Dataset Management/Data Ingestion safety audit:
  [docs/dataset_data_ingestion_safety_audit.md](dataset_data_ingestion_safety_audit.md).
- Routed upload, linked registration, transient-save, annotation metadata, and
  labelmap metadata updates through the strict guarded dataset metadata writer
  instead of the permissive helper that only logged write failures.
- Added rollback on metadata-write failure for backend-owned dataset zip
  uploads, linked dataset registration, and transient-session saves, preventing
  half-created library entries after a failed durable metadata write.
- Added regressions for upload/register/transient-save metadata failures.
- Validation: focused metadata rollback regressions (`3 passed`), affected
  dataset files (`101 passed`), focused dataset/data-ingestion safety bundle
  (`150 passed, 9 skipped`), live browser dataset/ingestion E2E (`9 passed`),
  `py_compile localinferenceapi.py tests/test_dataset_zip_upload_security.py
  tests/test_dataset_linked_annotation_flows.py`, `node --check
  ybat-master/ybat.js`, `git diff --check`, live UI endpoint map check with no
  missing paths or method mismatches, live OpenAPI sanity (`tested=144`,
  `failures=[]`), and the full pytest suite (`1201 passed, 19 skipped`) passed
  against the restarted backend.

## 2026-05-24: SAM3 and Qwen Dataset Metadata Persistence

- Made the shared dataset, Qwen dataset, and SAM3 dataset metadata persistence
  helpers fail loudly by default instead of logging write failures and returning
  success.
- Kept passive read/list metadata normalization best-effort by explicitly
  suppressing those write-back failures in dataset listing, glossary reads,
  Qwen-from-YOLO source glossary reads, RF-DETR dataset resolution, and
  segmentation-result metadata reads.
- Left conversion/materialization mutation paths strict, so YOLO-to-COCO,
  Qwen-to-COCO, COCO-to-YOLO, SAM3 annotation-overlay materialization, and SAM3
  split writes cannot report success without their final metadata record.
- Added regressions for dataset/Qwen/SAM3 metadata write failures and a
  YOLO-to-COCO conversion whose `sam3_dataset.json` cannot be written.
- Validation: `py_compile services/datasets.py localinferenceapi.py
  tests/test_dataset_metadata_io.py`, focused metadata IO (`11 passed`),
  affected dataset/Qwen/SAM3 backend bundle (`184 passed`), and the full pytest
  suite (`1206 passed, 19 skipped`) passed. The restarted backend also passed
  the live UI endpoint map check with no missing paths or method mismatches,
  live OpenAPI sanity (`tested=144`, `failures=[]`), and
  `/data_ingestion/capabilities` returned `local_salad` with MLX C-RADIO
  available.

## 2026-05-24: Annotation Snapshot In-Flight Save Race

- Fixed browser annotation snapshot flushing so a successful response only clears
  a dirty image key when the current browser snapshot still matches the payload
  that was sent.
- Added a queued follow-up flush for edits made while a snapshot request is in
  flight, preventing later local changes from losing their dirty flag after an
  older save completes.
- Added a browser regression that holds the first snapshot request, edits the
  same caption again, releases the request, and verifies the backend manifest
  contains the newer caption.
- Validation: `node --check ybat-master/ybat.js`, `py_compile` for the changed
  tests, focused static/backend annotation coverage (`17 passed`), targeted
  browser regression (`1 passed`), and the full dataset annotation browser file
  (`8 passed`) passed. The full pytest suite also passed
  (`1207 passed, 20 skipped`).

## 2026-05-24: Linked Dataset Export Labelmap Overrides

- Made dataset ZIP export include linked-dataset registry labelmap overrides as
  `labelmap.txt`, so downloads stay consistent with saved linked-dataset class
  edits without mutating the user's source dataset.
- Kept source files skipped when an override uses the same archive path, matching
  existing label/text overlay behavior.
- Rejected symlinked registry labelmap overrides during export instead of
  silently falling back to the source labelmap.
- Validation: `py_compile localinferenceapi.py
  tests/test_dataset_linked_annotation_flows.py`, focused dataset export
  regressions (`8 passed`), broader export/archive bundle (`110 passed`), and
  the full pytest suite (`1209 passed, 20 skipped`) passed. The restarted
  backend also passed the live UI endpoint map check with no missing paths or
  method mismatches, live OpenAPI sanity (`tested=144`, `failures=[]`), and
  `/data_ingestion/capabilities` returned `local_salad` with MLX C-RADIO
  available.

## 2026-05-24: Late-Cancel Finalization Guards

- Hardened auto-label finalization so a cancellation observed immediately before
  `save_dataset_annotation_snapshot` prevents the pending image write instead of
  committing labels after the user has cancelled.
- Moved auto-label per-class counters to update only after the snapshot save
  succeeds, and treat unserializable kept candidates as zero-write images instead
  of writing a no-op snapshot.
- Hardened data ingestion finalization so a cancellation observed after
  candidate/reference encoding but before result publication leaves no
  `result.json` or embeddings cache, and a cancellation observed after local
  SALAD training but before profile finalization writes no local head.
- Validation: `py_compile localinferenceapi.py
  tests/test_data_ingestion.py tests/test_auto_labeling_runner.py`, auto-label
  smoke regressions (`17 passed`), data-ingestion regressions (`45 passed`), and
  the dataset-management/data-ingestion safety bundle (`127 passed, 10 skipped`)
  passed. The full pytest suite also passed (`1213 passed, 20 skipped`). The
  restarted backend also passed the live UI endpoint map check with no missing
  paths or method mismatches, live OpenAPI sanity (`tested=144`, `failures=[]`),
  and `/data_ingestion/capabilities` returned the local-SALAD reference-profile
  flow with MLX C-RADIO available.

## 2026-05-24: Training Late-Cancel Publish Guards

- Extended the late-cancel finalization rule to CLIP classifier, Qwen, YOLOv8,
  and RF-DETR training workers.
- If cancellation is observed after the trainer returns but before durable
  metadata/artifact publication, the job now ends as `cancelled` without
  publishing classifier artifacts, Qwen run metadata, detector best weights,
  metrics JSON, pruning side effects, optimized RF-DETR exports, or success
  payloads.
- Added regressions for each worker so artifact/metadata publish helpers fail
  the test if they run after a late cancellation.
- Validation: `py_compile localinferenceapi.py
  tests/test_backend_job_start_validation.py tests/test_qwen_mlx_runtime.py
  tests/test_detector_active_lifecycle.py`, focused late-cancel regressions
  (`4 passed`), job/Qwen/detector suites (`116 passed`), dataset/data-ingestion
  safety bundle (`148 passed`), and the full pytest suite
  (`1217 passed, 20 skipped`) passed. The restarted backend also passed the
  live UI endpoint map check with no missing paths or method mismatches, live
  OpenAPI sanity (`tested=144`, `failures=[]`), and
  `/data_ingestion/capabilities` returned the local-SALAD reference-profile
  flow with MLX C-RADIO available.

## 2026-05-24: CLIP Classifier Rename Symlink Guards

- Audited the CLIP classifier registry mutation surface after the dataset/data
  ingestion pass and found the rename endpoint had no direct regressions.
- Changed classifier rename target selection to treat an existing symlink leaf
  as an occupied name instead of resolving through it. A requested rename such
  as `alias.pkl` where `alias.pkl` is a symlink now selects a safe suffixed
  destination like `alias_1.pkl` inside the classifier registry, leaving the
  symlink untouched.
- Added regressions for normal file/meta rename with active-classifier state
  update and for the symlink-target case.
- Validation: `py_compile localinferenceapi.py
  tests/test_clip_registry_downloads.py`, focused rename regressions
  (`2 passed`), CLIP registry coverage (`26 passed`), prepass/EDR artifact
  package coverage (`79 passed`), agent cascade import/export coverage
  (`30 passed`), and the full pytest suite (`1219 passed, 20 skipped`) passed.
  The restarted backend also passed the live UI endpoint map check with no
  missing paths or method mismatches, live OpenAPI sanity (`tested=144`,
  `failures=[]`), and a live `/clip/classifiers` registry read.

## 2026-05-24: SAM3 Run Root Symlink Guards

- Audited SAM3 training run deletion and cache cleanup after the classifier
  registry pass.
- Hardened the shared SAM3 run lookup helper to reject a symlinked job root or
  symlinked job-root parent before resolving a run id. This prevents scoped
  `delete_sam3_run` operations from deleting run directories through a
  redirected `SAM3_JOB_ROOT`.
- Made SAM3 run listing return no runs when the job root is symlinked, matching
  the fail-closed behavior used for deletion.
- Added regressions for symlinked SAM3 job roots and symlinked job-root parents.
- Validation: `py_compile services/sam3_runs.py
  tests/test_backend_path_containment.py`, focused SAM3 root regressions
  (`2 passed`), SAM3 active lifecycle/path-containment coverage (`57 passed`),
  and the full pytest suite (`1221 passed, 20 skipped`) passed. The restarted
  backend also passed the live UI endpoint map check with no missing paths or
  method mismatches, live OpenAPI sanity (`tested=144`, `failures=[]`), and a
  live `/sam3/models` registry read.

## 2026-05-24: Calibration Report Bundle Job Scoping

- Audited the calibration report bundle endpoint, which reads durable job
  artifacts and live-job result paths.
- Tightened report lookup from calibration-root containment to requested-job
  containment. A `report_bundle.json` symlink inside one calibration job can no
  longer redirect reads to a sibling job under the same calibration root, and a
  live job result cannot point `report_bundle_json` at another job's report.
- Added regressions for persisted symlink redirection and live-job sibling
  result paths.
- Validation: `py_compile localinferenceapi.py
  tests/test_calibration_report_bundle_endpoint.py`, focused report-bundle
  regressions (`2 passed`), calibration report/job-start coverage (`50 passed`),
  calibration helper/registry/resilience coverage (`44 passed`), and the full
  pytest suite (`1223 passed, 20 skipped`) passed. The restarted backend also
  passed the live UI endpoint map check with no missing paths or method
  mismatches, live OpenAPI sanity (`tested=144`, `failures=[]`), and a live
  `/calibration/jobs` read.

## 2026-05-24: Linked Dataset Listing Is Read-Only

- Re-audited Dataset Management and Data Ingestion from the data-loss angle,
  especially linked datasets where source files are user-owned.
- Found that dataset listing could still trigger read-time Qwen/COCO conversion
  or SAM3/glossary metadata backfills against a linked source root if a linked
  record pointed at a Qwen/COCO/SAM3-shaped tree. Normal UI registration is
  strict YOLO, but the backend invariant must hold for legacy or direct API
  records too.
- Changed linked dataset discovery so mutable discovery work stays in the
  backend registry overlay. Linked source roots are not auto-converted, not
  SAM3 metadata-backfilled, and not used for write-capable glossary discovery
  during listing.
- Added a regression that makes linked listing fail if source conversion,
  source metadata backfill, or source glossary backfill is attempted.
- Validation: `py_compile services/datasets.py
  tests/test_dataset_linked_root_status.py`, focused linked-listing regression
  (`1 passed`), dataset/data-ingestion safety bundle (`166 passed`), skipped UI
  E2E dataset checks in this environment (`10 skipped`), and the full pytest
  suite (`1224 passed, 20 skipped`) passed. The restarted backend also passed
  the live UI endpoint map check with no missing paths or method mismatches,
  live OpenAPI sanity (`tested=144`, `failures=[]`), and live `/datasets` plus
  `/data_ingestion/capabilities` reads.

## 2026-05-24: Dataset and Ingestion Rollback Cleanup

- Continued the Dataset Management/Data Ingestion data-loss audit, focusing on
  cleanup paths that run after a failed mutation or failed job start.
- Replaced raw `rmtree()` rollback cleanup for linked dataset registration and
  transient-session saves with the guarded dataset delete helper, so cleanup
  revalidates the registry root and refuses to delete through a symlinked
  parent.
- Added a data-ingestion job cleanup helper that revalidates the ingestion root,
  unlinks a symlinked job leaf without touching its target, and refuses cleanup
  if the job parent has become symlinked before the failure handler runs.
- Added regressions for registry-root swaps during register/transient rollback,
  ingestion-root swaps during analysis startup failure, and symlinked ingestion
  job cleanup.
- Validation so far: `py_compile localinferenceapi.py
  tests/test_data_ingestion.py tests/test_dataset_linked_annotation_flows.py`,
  `git diff --check`, focused cleanup/security regressions (`64 passed`), and
  the affected dataset/data-ingestion/Qwen safety bundle (`187 passed`) passed.
  The full pytest suite also passed with `1228 passed, 20 skipped`. The
  restarted backend passed `tools/check_ui_endpoints.py` with no missing paths or
  method mismatches, endpoint map check (`missing=[]`), endpoint method check
  (`failures=[]`), OpenAPI sanity (`tested=144`, `failures=[]`), and live
  `/datasets`, `/datasets/trash`, and `/data_ingestion/capabilities` reads.

## 2026-05-24: Linked Dataset Allowlist Revalidation

- Continued the Dataset Management/Data Ingestion audit from the legacy-record
  angle. Normal UI registration already required linked source roots to sit under
  `DATASET_LINK_ROOTS`, but hand-edited or stale registry records could still list
  an existing linked root after the allowlist changed.
- Dataset listing now revalidates linked roots against the current allowlist.
  Non-allowlisted roots are marked `not_allowlisted`, are not inspected for YOLO,
  caption, glossary, Qwen, or COCO readiness, and therefore cannot silently expose
  source files through discovery.
- Backend use-sites now reject linked records with `missing`, `invalid`, or
  `not_allowlisted` status before export, dataset checks, annotation access, or
  Data Ingestion reference media collection can touch the source tree.
- Validation so far: `py_compile localinferenceapi.py services/datasets.py
  tests/test_dataset_linked_root_status.py
  tests/test_dataset_linked_annotation_flows.py tests/test_data_ingestion.py`,
  `git diff --check`, the dataset/data-ingestion safety bundle (`173 passed`),
  the adjacent dataset/Qwen safety bundle (`229 passed`), and the full pytest
  suite (`1231 passed, 20 skipped`) passed. The restarted backend passed
  `tools/check_ui_endpoints.py` with no missing paths or method mismatches,
  endpoint map check (`missing=[]`), endpoint method check (`failures=[]`),
  OpenAPI sanity (`tested=144`, `failures=[]`), and live `/datasets`,
  `/datasets/trash`, and `/data_ingestion/capabilities` reads.

## 2026-05-24: Prepass Recipe ID Alias Hardening

- Continued the backend artifact-lifecycle audit outside Dataset Management and
  Data Ingestion, focusing on saved prepass recipes that can be exported,
  bundled into cascades, or deleted from the UI.
- Found that existing-recipe paths reused the create-time sanitizer. A malformed
  request such as `recipe a` could silently resolve to an existing `recipe-a`
  directory, allowing get/export/delete/save calls to touch the wrong recipe
  artifact.
- Prepass recipe IDs are now exact identifiers at the shared path resolver:
  callers must pass an already-safe ID, and sanitized aliases are rejected with
  `prepass_recipe_path_invalid` before any recipe metadata, zip, or directory is
  read or removed.
- Added regressions proving malformed aliases cannot delete, read, export, or
  create a sanitized recipe directory.
- Validation so far: `py_compile services/prepass_recipes.py
  tests/test_prepass_recipe_config_validation.py`, focused prepass recipe
  coverage (`38 passed`), the prepass/agent-recipe/cascade bundle (`96 passed`),
  `git diff --check`, and the full pytest suite (`1234 passed, 20 skipped`)
  passed. The restarted backend passed `tools/check_ui_endpoints.py` with no
  missing paths or method mismatches, endpoint map check (`missing=[]`),
  endpoint method check (`failures=[]`), OpenAPI sanity (`tested=144`,
  `failures=[]`), live `/datasets`, `/datasets/trash`, and
  `/data_ingestion/capabilities` reads, and a live malformed prepass lookup
  returned `400 prepass_recipe_path_invalid`.

## 2026-05-24: Agent Recipe and Cascade ID Path Hardening

- Continued from the prepass recipe ID alias fix into adjacent Agent Mining
  recipe and cascade artifacts.
- Found the same resolved-path-only pattern on agent recipe and cascade
  load/delete/zip-build paths. A malformed ID such as `alias_parent/../victim`
  could resolve to an existing direct-child artifact when the intermediate
  directory existed, making the request path touch the wrong saved recipe or
  cascade.
- Added explicit direct-child ID validators for Agent Mining recipes and
  cascades. IDs must now be ASCII safe names using only letters, digits, `.`,
  `_`, and `-`; slashes, backslashes, path aliases, empty IDs, and sanitizer
  aliases fail with the existing `agent_recipe_path_invalid` or
  `agent_cascade_path_invalid` errors before any JSON, zip, or recipe directory
  is read, built, or removed.
- Added regressions proving malformed recipe/cascade aliases cannot load or
  delete existing artifacts, and nested IDs cannot cause zip builders to create
  new parent directories.
- Validation so far: `py_compile services/prepass_recipes.py
  services/agent_cascades.py tests/test_prepass_recipe_config_validation.py
  tests/test_agent_cascade_export_safety.py`, focused recipe/cascade coverage
  (`60 passed`), the broader prepass/agent-recipe/cascade bundle (`104 passed`),
  `git diff --check`, and the full pytest suite (`1238 passed, 20 skipped`)
  passed. The restarted backend passed `tools/check_ui_endpoints.py` with no
  missing paths or method mismatches, endpoint map check (`missing=[]`),
  endpoint method check (`failures=[]`), OpenAPI sanity (`tested=144`,
  `failures=[]`), live `/datasets`, `/datasets/trash`, and
  `/data_ingestion/capabilities` reads, plus live malformed Agent Mining recipe
  and cascade lookups returning `400 agent_recipe_path_invalid` and
  `400 agent_cascade_path_invalid`.

## 2026-05-24: Dataset/Ingestion Transient Safety Recheck and CLIP Artifact Alias Hardening

- Re-ran the Dataset Management and Data Ingestion data-loss audit across UI
  actions and backend mutation paths: upload/register/open/save/delete/restore,
  annotation snapshot/meta save, linked export, profile build, candidate
  analysis, backend reference reads, and cancel.
- Found one transient linked-dataset read gap: `open_path` annotation manifests
  followed source `labels/*.txt` directly, while persisted linked manifests used
  guarded root-contained reads. Transient manifests now use the same guarded
  reader, so a symlinked source label that resolves outside the allowlisted
  dataset root is ignored instead of being exposed in the UI.
- Tightened Data Ingestion cancel UX so the Cancel button checks non-OK backend
  responses and disables repeat clicks while cancellation is being posted.
- Continued artifact-lifecycle hardening for CLIP classifiers and labelmaps.
  Deletion/path resolution now rejects path-alias components like
  `alias_parent/../target.pkl` before resolving to a real file, preventing
  malformed relative paths from touching an existing artifact.
- Added regressions for transient source-label symlink escapes and CLIP
  classifier/labelmap path-alias deletes while preserving plain nested relative
  artifact paths.
- Validation: `py_compile localinferenceapi.py services/classifier.py
  tests/test_dataset_linked_annotation_flows.py tests/test_clip_registry_downloads.py`,
  focused transient/CLIP regressions (`4 passed`), the dataset/data-ingestion
  safety bundle (`215 passed`), and the full pytest suite (`1243 passed, 20
  skipped`) passed. With `RUN_UI_E2E=1`, the live Dataset Management/Data
  Ingestion UI smoke tests (`2 passed`) and dataset annotation UI tests (`8
  passed`) also passed against the restarted backend and local UI server. The
  restarted backend passed `tools/check_ui_endpoints.py` with no missing paths or
  method mismatches, endpoint map check (`missing=[]`), endpoint method check
  (`failures=[]`), OpenAPI sanity (`tested=144`, `failures=[]`), and live
  `/datasets`, `/datasets/trash`, and `/data_ingestion/capabilities` reads.

## 2026-05-24: Final Inference Marker and Dataset Discovery Fail-Closed Pass

- Finished the current narrow hardening pass before pausing manual-audit work.
- Hardened the raw detector active-marker loader used by inference activation
  cleanup/status paths so symlinked `active.json` files, symlinked marker
  parents, and non-file marker targets are ignored instead of followed.
- Added service-level fail-closed guards to dataset discovery. The shared
  dataset-listing helper now skips registry/SAM3/Qwen dataset roots when the root
  or any parent component is a symlink, and skips child records with symlink
  components before reading metadata.
- Updated the Dataset/Data Ingestion safety audit to record the stronger
  discovery invariant.
- Validation: `py_compile localinferenceapi.py services/datasets.py
  tests/test_dataset_linked_root_status.py tests/test_detector_active_lifecycle.py`,
  focused symlink regressions (`3 passed`), the Dataset Management/Data
  Ingestion safety bundle (`176 passed`), and the inference artifact lifecycle
  bundle covering detector, SAM3, Qwen, and CLIP active/download/delete paths
  (`115 passed`) passed. The full pytest suite passed with `1246 passed, 20
  skipped`. After backend restart, `tools/check_ui_endpoints.py` reported no
  missing paths or method mismatches, endpoint map check returned `missing=[]`,
  endpoint method check returned `failures=[]`, OpenAPI sanity returned
  `tested=144`, `failures=[]`, and live reads of `/datasets`, `/datasets/trash`,
  `/data_ingestion/capabilities`, `/yolo/runs`, `/rfdetr/runs`, and
  `/sam3/storage/runs` succeeded. With `RUN_UI_E2E=1`, the Dataset
  Management/Data Ingestion UI smoke tests passed (`2 passed`).

## 2026-05-24: Inference Documentation Tidy Before Pause

- Rechecked the product docs around Qwen/MLX, detector, local SALAD, and
  hardening summaries before pausing this hardening cycle for manual testing.
- Found one stale benchmark-helper recommendation that still suggested reusing
  a locally trained SALAD head for crop-level Class Split or auto-class work.
  That conflicted with the current product decision and backend/UI enforcement:
  local SALAD remains limited to whole-image Data Ingestion diversity scoring.
- Removed the stale Class Split/auto-class local-SALAD recommendation from
  `tools/benchmark_salad_diversity.py`; the helper now reports the pooled crop
  baseline for class separation while keeping local SALAD recommendations in
  the Data Ingestion recipe section only.
- Added a contract test so future benchmark-helper edits cannot reintroduce
  crop-level local-SALAD guidance without failing the focused UI/product
  contract suite.
- Validation: `py_compile tools/benchmark_salad_diversity.py
  tests/test_labeling_panel_layout_contract.py`, focused local-SALAD UI/product
  contract coverage (`3 passed`), `git diff --check`, and the full pytest suite
  (`1247 passed, 20 skipped`) passed.

## 2026-05-24: Crop-Level SALAD Tooling Closure

- Found that the public backend/UI had already disabled crop-level
  `local_salad`, but direct auto-class training helpers and the class-separation
  benchmark still exposed stale crop-level SALAD paths.
- Made `tools/clip_training.py` fail closed with
  `local_salad_auto_class_disabled` before dataset or artifact access if direct
  callers request local SALAD aggregation.
- Simplified `tools/train_clip_regression_from_YOLO.py` so its public
  `--embedding-aggregation` choices only advertise the supported pooled mode.
- Retired local SALAD training/head flags from
  `tools/benchmark_salad_class_separation.py`; the helper now benchmarks only
  supported pooled DINOv3/C-RADIO Class Split recipes and records that
  crop-level local SALAD is disabled.
- Extended product-contract coverage to keep benchmark and direct-training
  helper outputs aligned with the Data Ingestion-only SALAD decision.
- Marked `docs/test_matrix.md` as an archived February validation scratchpad so
  old partial/failure rows are not mistaken for the current open-issue list.
- Validation: `py_compile tools/clip_training.py
  tools/train_clip_regression_from_YOLO.py
  tools/benchmark_salad_class_separation.py
  tests/test_clip_training_artifact_publish.py
  tests/test_labeling_panel_layout_contract.py`, focused crop-level SALAD
  rejection and product-contract coverage (`4 passed`), CLI help checks for the
  auto-class trainer and class-separation benchmark, `git diff --check`, and the
  full pytest suite (`1247 passed, 20 skipped`) passed.

## 2026-05-24: Qwen Training Fallback Catalog Alignment

- Rechecked Qwen runtime/model endpoints and UI endpoint coverage against the
  live backend. `/qwen/status`, `/qwen/models`, and `/qwen/settings` responded,
  and `tools/check_ui_endpoints.py http://127.0.0.1:8000` reported no missing
  paths or method mismatches.
- Found that the Qwen training UI fallback catalog was still a small stale
  CUDA-oriented subset. If the live `/qwen/models` registry fetch failed, the UI
  could hide MLX training presets and most abliterated choices even though the
  backend supports them.
- Rebuilt the fallback list with explicit metadata for CUDA, quantized CUDA,
  MLX-VLM, CUDA abliterated, quantized abliterated, and MLX abliterated
  training presets.
- Added a product-contract test to require MLX, abliterated, and quantized
  abliterated fallback entries plus their runtime/training-base metadata.
- Validation: `py_compile tests/test_labeling_panel_layout_contract.py`,
  focused Qwen fallback and local-SALAD UI contract coverage (`2 passed`),
  `tools/check_ui_endpoints.py http://127.0.0.1:8000`, `git diff --check`, and
  the full pytest suite (`1248 passed, 20 skipped`) passed.

## 2026-05-24: Qwen Runtime Select MLX Resolver Alignment

- Audited the remaining Qwen caption/prepass runtime model controls after the
  training fallback update.
- Found that recipe-loaded explicit model IDs were injected through
  `ensureQwenSelectOption`, but that helper only marked `mlx-community/*`
  models as MLX. Compatible abliterated MLX IDs from other namespaces, such as
  `*-mlx` and `*-mlx-*`, were therefore tagged as Transformers in the injected
  option even though the runtime resolver treats them as MLX-VLM.
- Routed injected Qwen select options through the shared
  `inferQwenRuntimePlatform` resolver so recipe-loaded MLX IDs use the same
  platform detection as training presets and runtime controls.
- Added a UI contract test requiring the shared resolver path and the non-
  `mlx-community` MLX heuristics used for abliterated model IDs.
- Validation: `py_compile tests/test_labeling_panel_layout_contract.py`,
  focused Qwen runtime-select contract coverage (`2 passed`),
  `tools/check_ui_endpoints.py http://127.0.0.1:8000`, `git diff --check`, and
  the full pytest suite (`1249 passed, 20 skipped`) passed.

## 2026-05-24: Qwen Caption Default Model Safety

- Continued the Qwen inference-control sweep after aligning injected runtime
  platform detection.
- Found that the captioning model selector defaulted to
  `Qwen/Qwen3-VL-30B-A3B-Thinking` instead of the active backend model. That
  made an ordinary captioning run unexpectedly choose the heaviest listed
  preset when the user had not made an explicit model choice.
- Changed the captioning selector to default to `Use active model`, keeping
  explicit 30B-A3B Thinking selection available but opt-in.
- Added a UI contract test so future changes keep exactly one default selection
  in the captioning model selector and keep it on the active runtime option.
- Validation: `py_compile tests/test_labeling_panel_layout_contract.py`,
  focused Qwen caption/runtime-select contract coverage (`2 passed`),
  `node --check ybat-master/ybat.js`,
  `tools/check_ui_endpoints.py http://127.0.0.1:8000`, `git diff --check`, and
  the full pytest suite (`1250 passed, 20 skipped`) passed.

## 2026-05-24: Data Ingestion Profile and Accepted Export Flow

- Added reference-profile ZIP export/import for local SALAD reference heads.
  Bundles carry a manifest, profile payload, checksum file, reference
  fingerprint, base-encoder metadata, and original/import provenance.
- Hardened profile imports with zip traversal, symlink, compressed-size,
  uncompressed-size, bundle-version, checksum, and local-head metadata checks
  before copying a profile into the backend-owned head store.
- Added accepted-output review endpoints for completed Data Ingestion analysis
  jobs. Users can keep/reject ranked items, preview the resulting outputs, and
  download a ZIP without mutating the submitted candidate files.
- Added export shaping controls for original images, aspect-fit resize, stretch
  resize, center crop, and fixed-size tiling with explicit edge policy,
  overlap, size, and JPEG-quality settings.
- Stored a stable reference fingerprint when reference profiles are trained,
  added stable analysis item ids for review/export selection, and kept export
  source paths constrained to the ingestion job directory.
- Wired the Data Ingestion UI with reference-profile download/upload controls,
  per-item keep checkboxes, accepted-output preview thumbnails, and ZIP
  download controls.
- Updated README and the Dataset/Data Ingestion safety audit to document the
  portable reference-profile flow and read-only accepted-data export invariant.
- Validation: `py_compile localinferenceapi.py api/data_ingestion.py`,
  `node --check ybat-master/ybat.js`, focused reference-profile and
  accepted-export regressions (`5 passed`), focused Data Ingestion/UI contract
  coverage (`54 passed`), `tools/check_ui_endpoints.py http://127.0.0.1:8000`,
  `git diff --check`, and the full pytest suite (`1255 passed, 20 skipped`)
  passed against the restarted backend.

## 2026-05-24: Data Ingestion Deep Export and Candidate Review Hardening

- Re-audited the Data Ingestion analysis, accepted-export, media-staging, and
  backend-reference paths after adding reference-profile and accepted-ZIP flows.
- Rejected unsupported upload extensions before writing files, closed rejected
  upload handles, bounded generated media names, recorded file mtimes, and
  included mtime identity in reference fingerprints so changed same-name media
  invalidates profile provenance.
- Applied a backend video frame extraction safety cap even when the UI sends
  `max_frames_per_video=0`, and exposed the cap in capabilities.
- Rejected symlinked backend dataset roots and symlinked Data Ingestion storage
  roots before reference reads or result reads.
- Hardened accepted-export planning: explicit empty item/output selections stay
  empty, invalid selection shapes fail fast, center-crop bounds are recorded
  accurately, too-small `drop_partials` sources emit no tiles, target geometry is
  capped, duplicate ZIP paths are rejected, and every render revalidates the
  source file inside the job directory.
- Split public analysis results from internal result state. Public result items
  now omit source paths and expose candidate-thumbnail URLs; internal accepted
  export still uses guarded source paths.
- Clarified candidate review scoring by adding reference-novelty score,
  percentile, and raw novelty rank alongside the farthest-first keep rank. The
  UI labels this as reference novelty instead of a generic score and shows a
  400 px hover preview for candidate cards.
- Removed preview-tile checkboxes from the UI because selecting a paginated
  preview tile could accidentally restrict the final ZIP to only the first
  preview page. Candidate keep/reject checkboxes remain the accepted-data
  selection control.
- Updated README and the Dataset/Data Ingestion safety audit to record the
  stronger accepted-export, candidate-thumbnail, and public-result invariants.
- Validation: `py_compile localinferenceapi.py api/data_ingestion.py
  services/data_ingestion.py tests/test_data_ingestion.py
  tests/test_labeling_panel_layout_contract.py`, `node --check
  ybat-master/ybat.js`, focused Dataset/Data Ingestion coverage
  (`185 passed`), `tools/check_ui_endpoints.py http://127.0.0.1:8000`,
  OpenAPI route confirmation for candidate and accepted-export thumbnails, and
  the full pytest suite (`1269 passed, 20 skipped`) passed against the
  restarted backend.

## 2026-05-24: Annotation Image Value Metric

- Added an optional Label Images checkbox for `Show diversity metric / image
  value`. When enabled, changing images or editing boxes refreshes a compact
  image value score beside the image selector.
- Kept the metric client-side and class-coverage based: it scores the active
  image's bboxes against dataset-wide class counts, with higher values for
  boxes that add rare or missing classes. This is separate from Data Ingestion's
  SALAD/reference-novelty scoring.
- Added a testable browser/Node helper for bbox-bucket and YOLO-line counting,
  image value computation, and compact display formatting.
- Validation: `node --check ybat-master/annotation_diversity.js`, `node --check
  ybat-master/ybat.js`, `py_compile tests/test_annotation_diversity_metric.py
  tests/test_labeling_panel_layout_contract.py`, focused annotation/UI coverage
  (`22 passed`), `tools/check_ui_endpoints.py http://127.0.0.1:8000`,
  `git diff --check`, and the full pytest suite (`1272 passed, 20 skipped`)
  passed against the running backend.

## 2026-05-25: Chunked Current-Dataset Upload Sessions

- Replaced Data Ingestion's browser-only active-reference fallback with the
  Dataset Management upload path: browser-only Label Images workspaces are saved
  as managed backend datasets first, then profile/analysis jobs receive the
  resulting dataset id.
- Added `/datasets/upload_session/*` for large current-workspace uploads. The
  browser streams bounded batches instead of building one huge ZIP or appending
  thousands of reference files to a Data Ingestion multipart request.
- Added backend upload-session sidecars under
  `uploads/yolo_dataset_upload_sessions` so interrupted sessions are visible
  after backend restart. Sessions can be listed, inspected, finalized, or
  explicitly cancelled; cancel removes only that guarded staging directory.
- Hardened finalize semantics: incomplete sessions are rejected when the
  expected image count is known; successful finalize removes the sidecar, writes
  durable dataset metadata, and promotes the complete YOLO tree into
  `uploads/datasets`.
- Wired Data Ingestion cancel to abort an active current-dataset upload session
  before any SALAD job exists, then call the upload-session cancel endpoint.
- Guarded active Label Images reference reuse by registered image count: a
  linked backend dataset or cached active-reference upload is reused only when
  its backend count matches the currently open Label Images workspace, so stale
  3.7k-image references cannot stand in for a newly opened 9.5k-image set.
- Added regression coverage for chunked upload finalization, restart recovery,
  incomplete-finalize rejection, and cancel cleanup.
- Validation: `python3 -m py_compile api/datasets.py api/data_ingestion.py
  localinferenceapi.py`, `node --check ybat-master/ybat.js`,
  `tests/test_dataset_zip_upload_security.py tests/test_data_ingestion.py
  tests/test_labeling_panel_layout_contract.py tests/test_api_route_uniqueness.py`
  (`126 passed`), live upload-session start/batch/status/list/cancel smoke,
  `tools/check_ui_endpoints.py http://127.0.0.1:8000`, and `git diff --check`.

## 2026-05-26: MLX-DINOv3 Mac Encoder Path

- Added an optional Swift/MLX DINOv3 worker pinned to
  `vincentamato/MLXDINOv3` commit
  `3122d7905cca21012b4c249e8ddad19ff78f54bc`.
- Added a backend resolver for `DINOV3_BACKEND=auto|torch|mlx`. Auto selects
  MLX on Apple Silicon only when the worker and converted ViT checkpoint are
  present; otherwise it falls back to Torch/MPS before a job starts.
- Wired the resolver into Data Ingestion profile training, candidate/reference
  scoring, DINOv3 class-analysis encoding, active DINOv3 classifier loading,
  post-training DINOv3 classifier resume, and Train Class Predictor DINOv3
  feature extraction without mixing MLX and Torch embeddings inside one job.
- Added a per-job Data Ingestion media worker control, parallelized reference
  view augmentation, merged left/right augmented encoder passes into a single
  DINOv3/C-RADIO batch, and bounded local SALAD training views at `384px`
  before the encoder's native resize to reduce avoidable CPU/image I/O.
- Added training-signature reuse for reference profiles so repeated builds of
  the same reference dataset/settings can return the existing profile instead
  of launching another full augmented DINOv3/SALAD pass.
- Added a list endpoint for in-memory Data Ingestion jobs to make active
  profile/analysis jobs recoverable after a tab refresh.
- Added build/convert scripts for the Swift worker and converted model cache.
  The build script now creates the colocated `mlx.metallib` needed by SwiftPM
  MLX binaries.
- Validation: focused Python coverage for MLX-DINOv3 resolution and Data
  Ingestion/classification/training routing, Swift worker build, ViT-B conversion, real
  worker smoke (`cls_token [1,768]`, `patch_tokens [1,196,768]`), Torch parity
  check (`CLS cosine min 0.999994`, patch cosine min `0.999998`), and a
  synthetic 32-image throughput check showing MLX about `342-361 imgs/s` versus
  Torch/MPS about `180-212 imgs/s`.
- Final focused regression bundle:
  `tests/test_mlx_dinov3_backend.py tests/test_data_ingestion.py
  tests/test_class_analysis.py tests/test_macos_acceleration.py
  tests/test_clip_training_artifact_publish.py
  tests/test_labeling_panel_layout_contract.py` (`257 passed`), plus
  `python3 -m py_compile localinferenceapi.py api/data_ingestion.py`,
  `node --check ybat-master/ybat.js`, and `git diff --check`.

## 2026-05-26: Data Ingestion Upload Error Reporting

- Fixed the active Label Images reference upload path so Data Ingestion preserves
  the real shared Dataset Management upload failure instead of replacing it with
  the misleading "did not return a backend dataset" message.
- Added backend-unreachable copy for browser fetch failures. If `localhost:8000`
  is down, the Data Ingestion panel now tells the user to start the backend
  instead of implying that a completed dataset upload returned malformed data.
- Confirmed the current 9,526-image active-reference dataset was already
  registered as
  `data_ingestion_reference_current_label_images_dataset_9526`, no chunk upload
  sessions were left behind, and the matching reference profile is visible when
  the backend is running.
- Validation: `node --check ybat-master/ybat.js`, `git diff --check`,
  `tests/test_dataset_zip_upload_security.py`, focused Data Ingestion backend
  reference tests (`19 passed`), and live `/datasets`,
  `/datasets/upload_sessions`, and `/data_ingestion/capabilities` probes against
  the restarted backend.

## 2026-05-29: Class Split Review And MLX SAM1 Runtime

- Added an optional MLX SAM1 adapter for Apple Silicon annotation prompts.
  `SAM1_BACKEND=auto` uses the existing Torch/MPS SAM path unless converted MLX
  SAM weights and the `mlx-examples` Segment Anything package are configured;
  `SAM1_BACKEND=mlx` fails closed when those assets are unavailable.
- Exposed SAM1 backend preference, Torch device candidates, and MLX adapter
  availability in `/system/health_summary`.
- Hardened active Label Images Data Ingestion reference reuse so candidate
  analysis can use a selected reference profile's stored backend dataset handle
  after confirming the dataset still exists and the image count matches the open
  workspace.
- Added Class Split graph display filtering for likely wrong-class points,
  floating graph crop previews, optional cluster hull overlays, and a Cluster
  Proposals panel with representative crop, class mix, purity, mean outlier, and
  select-cluster bulk review behavior.
- Made cluster selection switch the graph back to all-object display before
  selecting every cluster member, so bulk relabeling cannot silently act on
  objects hidden by the wrong-only graph filter.
- Updated macOS runtime docs, README update tracking, Class Split/Data Ingestion
  flow docs, env examples, and UI contract tests for the new controls.
- Validation: `node --check ybat-master/ybat.js`, `py_compile
  localinferenceapi.py services/mlx_sam.py`, `git diff --check`, and focused
  Data Ingestion, Class Analysis, macOS acceleration, MLX-DINOv3, MLX-SAM, SAM
  preload, and UI contract coverage (`219 passed`, `8 warnings`).

## 2026-05-30: Dataset Upload Guardrails And Class Split Navigation

- Added a Dataset Management "Staged upload sessions" panel backed by the
  existing `/datasets/upload_sessions` and upload-session cancel endpoints, so
  interrupted browser uploads are visible and cancellable without touching
  registered datasets or source files.
- Hardened transient server-path dataset saves. After a transient session is
  saved into the library, the temporary path controls are cleared and the UI
  tells the user to reopen the dataset card before continuing persistent edits.
- Disabled source-dependent actions for unavailable linked datasets while
  keeping delete available. This prevents annotation, ingestion, download,
  conversion, or Qwen export actions from acting on missing source roots.
- Tightened active Label Images reference reuse in Data Ingestion by allowing
  unknown backend image counts but rejecting explicit mismatches, which avoids
  stale count assumptions blocking valid active-workspace reference profiles.
- Added a Class Split graph drag-mode selector (`lasso`, `box`, `pan`) and
  moved graph selection state onto Plotly `selectedpoints` plus
  `selectionrevision`, so lasso and cluster selections stay visually stable
  across rerenders and both normal/wrong-class traces dim consistently.
- Hardened Class Split jump-to-point behavior from wrong-only display: selecting
  an object outside the current wrong-only filter now restores all-object
  display before focusing/flashing the point.
- Hardened Train Class Predictor's `Test Size = 0` flow so explicit zero
  evaluation uses all samples for training instead of falling into sklearn split
  errors. The UI help now states the group-split, stratified fallback, and
  all-train behavior.
- Validation: `node --check ybat-master/ybat.js`, `git diff --check`, focused
  Class Split/auto-class tests (`3 passed`), broader class-analysis and
  auto-class bundle (`143 passed`, `1 skipped`), live
  `tools/check_ui_endpoints.py http://127.0.0.1:8000`, browser top-tab
  navigation E2E against the restarted backend (`1 passed`), and
  `/system/health_summary` returned `ok: true`.

## 2026-05-30: Named Active-Workspace Uploads For Ingestion And Class Split

- Added explicit upload-name controls for active Label Images uploads launched
  from Data Ingestion (`Current upload dataset name`) and Class Split Explorer
  (`Workspace upload name`). These requested ids are handed to the backend
  dataset registry, which still uniquifies them when needed, so temporary
  current-workspace datasets can be identified and deleted later from Dataset
  Management.
- Wired Class Split browser-only active workspaces into the shared
  `/datasets/upload_session/*` chunked upload path before analysis. Backend
  linked/transient workspaces still submit direct `/class_analysis/jobs`
  references; the old one-shot `/class_analysis/jobs/active_workspace`
  multipart path remains only as a fallback when chunked upload cannot be used.
- Preserved raw active-workspace YOLO label lines during Class Split-triggered
  uploads, so all-class runs include labels for images that are listed in the
  active dataset but have not been hydrated by the browser.
- Made the Data Ingestion and Class Split active-workspace upload caches include
  the requested dataset name. Changing the name now creates or resolves a
  distinct managed dataset instead of silently reusing a previous auto-named
  upload with matching images.
- Validation: `node --check ybat-master/ybat.js`, `git diff --check`, focused
  Data Ingestion/Class Split UI contract tests, chunked upload-session finalize
  regression, and active-workspace class-analysis manifest regression (`4
  passed`).

## 2026-06-04: Class Split Active Workspace Reverted To Transient Job State

- Removed the Class Split `Workspace upload name` control and the local-storage
  reuse cache for Class Split active-workspace uploads. Data Ingestion still
  uses named managed uploads because reference profiles are reusable artifacts;
  Class Split is a live audit of the currently open annotation workspace.
- Browser-only Class Split analysis now uses
  `/class_analysis/jobs/active_workspace` as a transient job-local workspace.
  The uploaded images and label lines live under the Class Split job directory
  and are not registered in Dataset Management.
- Class Split mobile review is desktop-workspace only. Mobile actions are
  appended to the live mobile-review session and later synced into the open
  Label Images workspace; they never write directly to a backend dataset
  snapshot.
- Mobile review can still read context crops from backend-linked, transient, or
  active-workspace Class Split sources. Active-workspace reads are restricted to
  the parent Class Split job directory and its manifest.
- Added stable mobile `action_id`/`sequence` values, session pruning, and an
  active-workspace mobile context regression. Also hardened desktop sync so
  class changes use the matched open bbox class rather than a stale graph class.
- Validation: `node --check ybat-master/ybat.js`, `python3 -m py_compile
  localinferenceapi.py api/class_analysis.py`, `git diff --check`,
  `tests/test_labeling_panel_layout_contract.py` (`22 passed`), and
  `tests/test_class_analysis.py` (`89 passed`).

## 2026-06-02: Class Split Graph Projection And UI State Closeout

- Added switchable Class Split graph projection modes: global PCA,
  class-balanced PCA, between-class PCA, within-filter PCA, and UMAP when
  available. Class-balanced PCA is the default all-class view; within-filter PCA
  is guarded so all-class overlays require a class filter instead of drawing a
  misleading global plot.
- Moved large PCA coordinate arrays out of the public Class Split result JSON.
  Completed jobs write `projection_coordinates.npz`, public results expose
  `coordinates_available`, and the UI lazily fetches coordinates through
  `/class_analysis/jobs/{job_id}/projection/{mode}`.
- Hardened Class Split graph state: one Plotly trace per visible class for
  class coloring, explicit graph status text, stale placeholder purging, render
  tokens for async projection loads, and previous-result restoration when
  upload or job start fails.
- Tightened review interactions: confirming likely-wrong objects prunes hidden
  wrong-only selections, cluster hulls enable only when visible subclass
  clusters can actually draw a hull, malformed bbox matches no longer coerce to
  zero, and legacy PCA results map to global PCA without breaking the graph.
- Validation: `node --check ybat-master/ybat.js`,
  `py_compile localinferenceapi.py api/class_analysis.py`, `git diff --check`,
  focused class-analysis and layout tests, API route/UI endpoint contracts, and
  live Playwright Class Split/navigation contracts against the local backend
  (`99 passed`, `1 skipped` across the closeout commands).

## 2026-06-02: Class Split Projection Re-Audit

- Re-reviewed the committed Class Split graph/projection path after the
  post-analysis blank-plot concern: setup and toolbar projection controls,
  result projection inference, lazy coordinate loading, filter/display pruning,
  placeholder purging, cluster-hull gating, likely-wrong vignettes, inspector
  context crops, and jump-to-source matching.
- Confirmed normal backend results still keep `points` in public
  `/class_analysis/jobs/{job_id}/result` payloads while storing only the extra
  switchable PCA coordinate matrices in `projection_coordinates.npz` for lazy
  fetch through the projection endpoint.
- Validation: `node --check ybat-master/ybat.js`, macOS venv
  `py_compile` for `localinferenceapi.py` and `api/class_analysis.py`,
  `tests/test_class_analysis.py -q` (`81 passed`, `1 skipped`), live
  `RUN_UI_E2E=1` Playwright Class Split contract against the existing local
  backend/UI (`14 passed`), `tools/check_ui_endpoints.py
  http://127.0.0.1:8000` reported no missing paths or method mismatches, local
  `/system/health_summary` returned `ok: true`, and `git diff --check` passed.

## 2026-06-02: Label Image Selection Load Hardening

- Re-reviewed the current Class Split projection/frontend and backend result
  paths, then hardened the Label Images local file-selection flow found during
  the pass.
- Local image selection now stages the file list and displays the first image
  before the slower dimension scan runs. Dimension indexing continues in the
  background with stale-scan cancellation and phase-aware progress updates, so
  bbox imports/crop jobs are not blocked by image metadata indexing.
- Local image decode now uses object URLs instead of `FileReader.readAsDataURL`
  for normal display and helper loading. The original `File` blob remains on
  each image record for Class Split uploads, crop export, and downstream
  packaging, while base64 remains available only when a later tool explicitly
  asks for it.
- Validation: `git diff --check`, `node --check ybat-master/ybat.js`,
  `py_compile localinferenceapi.py api/class_analysis.py`, and
  combined pytest coverage for `tests/test_labeling_panel_layout_contract.py`,
  `tests/test_data_ingestion.py`, and `tests/test_class_analysis.py`
  (`190 passed`, `2 skipped`; sandbox emitted a Metal-device warning at
  interpreter shutdown). Playwright Class Split contracts were skipped under
  the current non-UI test configuration.

## 2026-06-02: Class Split Subclass Search Closeout

- Removed browser-heavy cluster coloring and hull overlays from all-class Class
  Split graphs. All-class mode now stays focused on class overview,
  likely-wrong review, and dataset analysis.
- Added explicit selected-class subclass search with sensitivity, max-cluster,
  and min-size controls. The backend exposes separate cluster-search jobs that
  reuse the parent analysis embeddings and report progress through
  `/class_analysis/jobs/{job_id}/cluster_search`.
- Moved cluster proposals and graph report beneath the plot, kept selected crop
  preview at the top of the right stack, and changed likely-wrong review into a
  full-width 12-vignette queue with shuffle, skip, confirm, source jump, and
  explicit suggested-class relabel actions.
- Hardened frontend state so stale subclass-search polling cannot mutate a
  newer Class Split result. Class cycling in Label Images now keeps the `E/R`
  carousel light by rendering a bounded class window, syncing Qwen/SAM3 target
  selects without rebuilding full option lists, and debouncing Class Split
  control refreshes.
- Updated README, the Class Split/Data Ingestion flow review, shortcut
  explainer text, and frontend cache keys.
- Validation: `node --check ybat-master/ybat.js`,
  `py_compile api/class_analysis.py localinferenceapi.py`, `git diff --check`,
  focused class-analysis backend tests, and focused labeling-panel layout
  contract tests. The Playwright Class Split contract remains fixture-skipped
  unless run with the UI E2E environment enabled.

## 2026-06-03: Backend-Served UI And UMAP Subclass Controls

- Renamed the browser entrypoint from `ybat.html` to `tator.html`. The backend
  now serves the UI at `/` and `/tator.html`, serves the whitelisted static UI
  assets from `ybat-master/`, and redirects legacy `/ybat.html` requests to
  `/tator.html`.
- Updated the startup documentation so the daily command remains
  `tools/run_macos_backend.sh`, with the normal browser target now on the same
  backend port instead of a separate static server. Static serving on port 8080
  remains documented only as an optional frontend-development path.
- Added backend route-contract coverage for the served UI, asset whitelist, and
  legacy redirect, and moved frontend file-contract tests to read
  `ybat-master/tator.html`.
- Extended Class Split projection guidance and controls: UMAP is explained as
  the selected-class subclass-island view, UMAP neighbor/min-distance levers are
  exposed, and explicit subclass search can use UMAP-island proposals or strict
  embedding KMeans proposals.
- Validation: `py_compile localinferenceapi.py`, `node --check
  ybat-master/ybat.js`, `git diff --check`, focused route/layout/calibration UI
  contracts, focused class-analysis coverage, live backend checks for `/`,
  `/tator.html`, `/ybat.html`, `/ybat.js`, and `/system/health_summary`.

## 2026-06-08: Shortcut Remapping And Qwen Specificity Probe Closeout

- Replaced the static Label Images shortcut explainer with a registry-backed
  shortcut panel that lists the active bindings, stores browser-local
  customizations, and supports reset, clear, import, and export actions.
- Added remappable actions for next/previous image, next/previous class, start
  drawing, end drawing, cancel drawing/focus, box deletion, latest-box deletion,
  SAM point and multi-point controls, region-detect hold, Auto/SAM mode toggles,
  YOLO-caption export, SAM3 similarity, and direct class ID slots `0` through
  `19`.
- Kept the shortcut editor inside the Label Images sidebar instead of adding
  another top-level tab, and updated light, dark, and Pip-Boy theme styling so
  key chips and remap rows remain readable.
- Added the Class Split Qwen specificity probe and region-contrast evidence
  path: the review loop can render clean context, target-only pixels,
  target-removed context, and strongest-overlap pixels, then ask Qwen to
  separate target-specific evidence from background/overlap cues before final
  review.
- Extended benchmark audit output with specificity-probe status, margin,
  reconciliation, guarded signal strength, and confirm-current rebuttal checks.
- Validation: `node --check ybat-master/ybat.js`, `py_compile` for
  `localinferenceapi.py`, the Qwen benchmark tools, and the focused tests;
  `tests/test_labeling_panel_layout_contract.py` (`23 passed`),
  `tests/test_tator_ui_routes.py` (`1 passed`), and combined
  `tests/test_class_analysis.py tests/test_qwen_review_benchmark_audit.py`
  (`251 passed`). `git diff --check` passed. The in-app browser surface was not
  available in this Codex session, so browser verification was limited to route
  and static contract tests.

## 2026-06-12: Dataset Upload Session Start Validation

- Re-audited the backend and UI route sanity checks after the broad platform
  hardening pass. `tools/run_ui_openapi_sanity.py` found that
  `POST /datasets/upload_session/start` accepted an empty JSON body and created
  a persistent staged-upload session.
- Hardened staged dataset uploads so session creation now rejects malformed
  payloads before creating any session directory. A start request must include a
  non-empty dataset id or run name, a supported dataset type (`bbox` or `seg`),
  and a positive expected image count.
- Added a regression that proves an empty start payload returns a client error
  and leaves both the upload-session root and in-memory session registry
  untouched.
- Validation: `git diff --check`, `py_compile localinferenceapi.py`,
  `tests/test_dataset_zip_upload_security.py` (`18 passed`), full
  `pytest -q` (`1536 passed`, `39 skipped`), `tools/run_refactor_validation.sh`
  with fuzz skipped, `tools/check_ui_endpoints.py http://127.0.0.1:8000`,
  `tools/run_ui_openapi_sanity.py http://127.0.0.1:8000`,
  `tools/run_openapi_missing_query_sanity.py http://127.0.0.1:8000`,
  `tools/run_ui_smoke.py --base-url http://127.0.0.1:8000`, rendered
  Playwright control coverage, and full `RUN_UI_E2E=1` browser E2E
  (`41 passed`) against the restarted backend.

## 2026-06-12: UI Parameter Sweep Preconditions

- Re-ran the live UI concurrency, endpoint, contract, fuzz, and parameter-sweep
  checks against the running backend. The parameter sweep was incorrectly
  classifying detector `412 Precondition Failed` responses as regressions when
  no active YOLO or RF-DETR model was configured.
- Made `tools/run_ui_param_sweep.py` import-safe and gave it the same explicit
  negative-path contract used by the other UI validation tools. Optional
  detector routes may return `412` when their active model precondition is not
  satisfied; that is a clean validation result, not a product failure.
- Added regression coverage proving the sweep no longer performs backend calls
  during import and that detector precondition responses remain accepted.
- Validation: `py_compile tools/run_ui_param_sweep.py`,
  `tests/test_validation_cleanup_tools.py` (`7 passed`), and
  `tools/run_ui_param_sweep.py http://127.0.0.1:8000` (`failures: []`).
  Broader validation also passed: `git diff --check`, `node --check
  ybat-master/ybat.js`, UI endpoint method check (`276` fetches, no failures),
  UI contract tests (`82` checks, no failures), UI concurrency smoke, OpenAPI
  sanity (`167` tested, no failures), missing-query sanity (`5` tested, no
  failures), backend health summary (`ok: true`), and full `pytest -q`
  (`1538 passed`, `39 skipped`).

## 2026-06-12: Validation Script Python Resolution

- Continued the broad validation pass and found that
  `tools/run_refactor_validation.sh` failed on the current macOS shell because
  it called `python` directly. The project validation path should not depend on
  callers manually prepending `.venv-macos/bin` to `PATH`.
- Hardened `tools/run_refactor_validation.sh` and `tools/run_fuzz_fast.sh` so
  they resolve Python in a predictable order: explicit `PYTHON`, repo-local
  `.venv-macos`, repo-local `.venv`, then `python3`/`python`. Both scripts now
  anchor to the repository root before invoking project files.
- The refactor wrapper passes its resolved interpreter into the fuzz wrapper, so
  nested validation uses the same Python environment.
- Validation: `SKIP_GPU=1 BASE_URL=http://127.0.0.1:8000
  tools/run_refactor_validation.sh` completed pycompile and Tier-0/Tier-1 fuzz,
  and a direct `tools/run_fuzz_fast.sh` run completed with `skip_gpu: true`.

## 2026-06-12: Qwen Benchmark Script Portability

- Extended the script audit to the Qwen benchmark tooling and MLP automation.
  Both Qwen prepass shell wrappers lacked execute bits and failed through
  `bash` with `python: command not found`; `tools/auto_mlp_run.sh` also carried
  an old host-specific `/home/steph/Tator` checkout path.
- Made the Qwen prepass smoke and benchmark wrappers executable, repo-root
  anchored, and Python-resolving using the same explicit `PYTHON` override /
  repo venv / `python3` fallback policy as the validation scripts.
- Made `tools/auto_mlp_run.sh` derive its root from the script location unless
  `ROOT_DIR` is explicitly supplied, create its log directory, select
  `.venv`/`.venv-macos` when available, and run all Python steps through the
  resolved interpreter.
- Added regression coverage that the Qwen wrappers are executable and
  interpreter-resolving, and that the MLP runner has no host-specific root or
  bare Python invocations.
- Validation: shell syntax for all `*.sh` files, direct `--help` execution for
  `tools/run_qwen_prepass_smoke.sh` and `tools/run_qwen_prepass_benchmark.sh`,
  and `tests/test_validation_cleanup_tools.py` (`9 passed`). Broader validation
  also passed: `git diff --check`, `tools/run_refactor_validation.sh` with
  `SKIP_GPU=1`, UI endpoint method check (`276` fetches, no failures), UI
  contract tests (`82` checks, no failures), backend health summary
  (`ok: true`), and full `pytest -q` (`1540 passed`, `39 skipped`).

## 2026-06-12: Browser E2E Default Harness

- Rechecked the browser E2E command used in the hardening ladder and found that
  `RUN_UI_E2E=1 pytest tests/ui/e2e` still skipped most browser coverage unless
  `UI_PAGE_URL` and `UI_DATASET_PATH` were set manually.
- Updated the E2E environment helper to use the backend-served UI at
  `${UI_API_ROOT:-http://127.0.0.1:8000}/tator.html` by default and the
  repo-local `tests/fixtures/fuzz_pack` dataset by default. Explicit
  `UI_PAGE_URL`, `UI_DATASET_PATH`, staging, and API-root overrides are still
  honored.
- Added regression coverage for the default URL/dataset behavior and explicit
  override behavior.
- Validation: `tests/test_ui_e2e_env_defaults.py` (`2 passed`) and
  `RUN_UI_E2E=1 tests/ui/e2e` (`41 passed`) against the running backend.
  Broader validation also passed: `git diff --check`, `py_compile` for the E2E
  env helper, focused validation-tool tests (`11 passed`), `node --check
  ybat-master/ybat.js`, UI endpoint method check (`276` fetches, no failures),
  UI contract tests (`82` checks, no failures), backend health summary
  (`ok: true`), and full `pytest -q` (`1542 passed`, `39 skipped`).
