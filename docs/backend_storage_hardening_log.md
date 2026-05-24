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
