# Dataset and Data Ingestion Safety Audit

This note maps the Dataset Management and Data Ingestion surfaces that can
mutate, move, or remove user data to the backend guard and regression coverage.

## Dataset Management

| Flow | Data touched | Guard | Coverage |
|---|---|---|---|
| Upload YOLO zip | Creates a backend-owned dataset copy under `uploads/datasets` | Zip traversal/symlink/size checks, guarded registry child path, strict metadata write, rollback on metadata failure | `tests/test_dataset_zip_upload_security.py` |
| Upload current labeling session | Packages browser images/labels, then uses the same upload endpoint | UI geometry/labelmap validation, upload endpoint guards | `tests/ui/e2e/test_dataset_ingestion_safety_flows.py`, upload security tests |
| Register server path | Creates a linked registry record only | Link-root allowlist, strict YOLO shape/labelmap validation, guarded registry path, guarded rollback on metadata failure | `tests/test_dataset_linked_annotation_flows.py` |
| Open transient server path | Creates in-memory transient session only | Link-root allowlist, strict YOLO shape/labelmap validation, TTL expiry, source label/text reads stay root-contained and ignore symlink escapes | `tests/test_dataset_linked_annotation_flows.py`, UI E2E |
| List linked datasets | Reads linked source metadata and registry overlays | Dataset storage roots fail closed if the root or parent is a symlink; linked roots are rechecked against `DATASET_LINK_ROOTS`; non-allowlisted roots are marked unavailable and not inspected. No read-time source conversion or metadata backfill writes for linked roots; registry metadata remains the mutable overlay | `tests/test_dataset_linked_root_status.py` |
| Save transient to library | Persists linked registry metadata plus overlay labels/text | Guarded registry path, registry-only labelmap overlay, strict metadata write, guarded rollback on failure | `tests/test_dataset_linked_annotation_flows.py` |
| Delete linked dataset | Removes linked registry record and overlays only | Source root is never passed to delete helper; active annotation/job guards block delete | `tests/test_dataset_linked_annotation_flows.py` |
| Delete managed dataset | Moves backend-owned dataset to trash | Managed-root containment, symlink rejection, active annotation/job guards, rollback during trash metadata failure | `tests/test_dataset_linked_annotation_flows.py`, UI E2E |
| Restore managed dataset | Moves trash entry back into library | Trash-id validation, target-id uniqueness, symlink rejection, metadata rollback on failure | `tests/test_dataset_linked_annotation_flows.py`, UI E2E |
| Download dataset zip | Creates a transient archive for export | Source-root containment, overlay labels/text merged into normal YOLO paths, linked registry labelmap overrides source labelmap, symlinked overrides rejected | `tests/test_dataset_linked_annotation_flows.py`, `tests/test_dataset_download_cleanup.py` |
| Annotation snapshot/meta save | Writes backend overlay labels/text and metadata | Active-lock ownership, image existence checks, guarded overlay roots, atomic no-follow writes, in-flight browser save race protection | `tests/test_dataset_linked_annotation_flows.py`, `tests/ui/e2e/test_dataset_annotation_flows.py` |
| Dataset glossary save | Writes canonical glossary into backend metadata | Guarded metadata root and strict metadata write | `tests/test_dataset_linked_annotation_flows.py`, `tests/test_glossary_library.py` |
| Dataset conversion/materialization | Writes COCO sidecars plus SAM3/Qwen metadata for training/build views | Strict final metadata writes; read-time metadata backfills are best-effort only | `tests/test_dataset_metadata_io.py`, `tests/test_dataset_linked_annotation_flows.py` |

## Data Ingestion

| Flow | Data touched | Guard | Coverage |
|---|---|---|---|
| Build reference profile | Creates job staging data and a local reference-profile head | Guarded job root, upload size/quota checks, backend-reference source validation, guarded startup cleanup, strict local-head path checks, final cancellation check before head write | `tests/test_data_ingestion.py` |
| Import/export reference profile | Reads or creates a local reference-profile head bundle | Zip traversal/symlink/size checks, checksum verification, bundle-version validation, strict local-head path checks, preserved provenance metadata and reference fingerprint | `tests/test_data_ingestion.py` |
| Analyze candidates | Creates job staging data, result JSON, embeddings cache | Guarded job root, candidate/reference gating, matching reference-profile validation, guarded startup cleanup, guarded result reads, final cancellation check before result write | `tests/test_data_ingestion.py`, UI E2E |
| Preview/download accepted candidates | Creates preview thumbnails and a transient export ZIP | Completed-analysis gating, selected-item/output validation, source-path containment inside the job root, output-count limits, explicit crop/resize/tile policy, source files read-only | `tests/test_data_ingestion.py`, UI contract |
| Backend reference dataset use | Reads existing dataset images | Dataset id resolution, path containment inside selected dataset root, active-job delete blocking by dataset id | `tests/test_data_ingestion.py`, `tests/test_dataset_linked_annotation_flows.py` |
| Cancel job | Mutates in-memory job state | Terminal jobs are not cancellable; active jobs move through `cancelling`; UI checks cancel response and blocks duplicate cancel clicks | `tests/test_data_ingestion.py` and endpoint sanity |

## Current Invariant

Dataset Management should not report success for a library mutation unless the
durable metadata record was actually written. Upload, register, and transient
save paths now roll back their backend-created dataset directory if the metadata
write fails, avoiding half-created library records. Those rollback deletes also
revalidate the registry root at cleanup time and refuse to delete through a
symlinked parent. SAM3/Qwen dataset conversion
and materialized training-view writes now also fail the mutation when final
metadata cannot be written, while passive read/list metadata cleanup stays
best-effort so existing datasets remain browseable.

The browser annotation workspace must also keep local dirty state until the exact
snapshot that was saved still matches the current image state. If a user edits an
image while a snapshot request is in flight, the older successful response now
queues another save instead of clearing the newer edit's dirty flag.

Linked dataset exports must be self-consistent: registry-owned labelmap edits are
included in the downloaded `labelmap.txt`, matching the overlaid labels and text
labels, while the user's original linked source labelmap remains untouched.
Dataset listing follows the same rule: linked source roots are never auto-converted
or metadata-backfilled during read-time discovery. Any mutable state for a linked
dataset must live in the backend registry overlay. Linked records whose source
root is outside the current `DATASET_LINK_ROOTS` allowlist are treated as
unavailable; they are not inspected during listing and cannot be downloaded or
used as Data Ingestion reference datasets until re-registered under an allowed
root.

Data ingestion cancellation must stop finalization before creating new result
artifacts. If cancellation is observed after candidate/reference encoding but
before the analysis result write, no `result.json` or embeddings cache is
published. If cancellation is observed after local reference-profile training but
before the head write, no local SALAD head is added. Startup failure cleanup
also revalidates the data-ingestion root and refuses to remove a job directory
through a symlinked parent.

Reference profile bundles are portable backend-owned artifacts, not source
dataset snapshots. Exported bundles include a manifest and checksums; imported
bundles are staged, validated, and copied into the local SALAD head store only
after their metadata and payload integrity checks pass.

Accepted-data exports are also read-only with respect to candidate sources. The
review UI records which analysis items are kept, the backend verifies that each
source path still resolves inside the ingestion job directory, and preview or
download outputs are rendered into backend-owned thumbnail/ZIP artifacts. This
flow does not move, overwrite, resize in place, or delete the submitted images
or extracted video frames.

The same late-cancel boundary now applies to adjacent training jobs that create
dataset-derived artifacts. CLIP classifier, Qwen, YOLOv8, and RF-DETR workers
re-check cancellation after trainer return and before durable publication, so a
cancelled job cannot publish new model metadata or success payloads after the
user has asked it to stop.

Transient linked-dataset manifests now use the same guarded source-label read
path as persisted linked annotation manifests. A symlinked `labels/*.txt` source
file that resolves outside the allowlisted dataset root is ignored instead of
being followed into external content.

Dataset discovery also fails closed at the shared service helper when a runtime
storage root or any of its parents has become a symlink. This keeps a swapped
`uploads/datasets`, SAM3 dataset root, or Qwen dataset root from being listed as
trusted backend-owned data.
