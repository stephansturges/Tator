# Dataset and Data Ingestion Safety Audit

This note maps the Dataset Management and Data Ingestion surfaces that can
mutate, move, or remove user data to the backend guard and regression coverage.

## Dataset Management

| Flow | Data touched | Guard | Coverage |
|---|---|---|---|
| Upload YOLO zip | Creates a backend-owned dataset copy under `uploads/datasets` | Zip traversal/symlink/size checks, guarded registry child path, strict metadata write, rollback on metadata failure | `tests/test_dataset_zip_upload_security.py` |
| Upload current labeling session | Packages browser images/labels, then uses the same upload endpoint | UI geometry/labelmap validation, upload endpoint guards | `tests/ui/e2e/test_dataset_ingestion_safety_flows.py`, upload security tests |
| Register server path | Creates a linked registry record only | Link-root allowlist, strict YOLO shape/labelmap validation, guarded registry path, rollback on metadata failure | `tests/test_dataset_linked_annotation_flows.py` |
| Open transient server path | Creates in-memory transient session only | Link-root allowlist, strict YOLO shape/labelmap validation, TTL expiry | `tests/test_dataset_linked_annotation_flows.py`, UI E2E |
| Save transient to library | Persists linked registry metadata plus overlay labels/text | Guarded registry path, registry-only labelmap overlay, strict metadata write, rollback on failure | `tests/test_dataset_linked_annotation_flows.py` |
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
| Build reference profile | Creates job staging data and a local reference-profile head | Guarded job root, upload size/quota checks, backend-reference source validation, strict local-head path checks, final cancellation check before head write | `tests/test_data_ingestion.py` |
| Analyze candidates | Creates job staging data, result JSON, embeddings cache | Guarded job root, candidate/reference gating, matching reference-profile validation, guarded result reads, final cancellation check before result write | `tests/test_data_ingestion.py`, UI E2E |
| Backend reference dataset use | Reads existing dataset images | Dataset id resolution, path containment inside selected dataset root, active-job delete blocking by dataset id | `tests/test_data_ingestion.py`, `tests/test_dataset_linked_annotation_flows.py` |
| Cancel job | Mutates in-memory job state | Terminal jobs are not cancellable; active jobs move through `cancelling` | `tests/test_data_ingestion.py` and endpoint sanity |

## Current Invariant

Dataset Management should not report success for a library mutation unless the
durable metadata record was actually written. Upload, register, and transient
save paths now roll back their backend-created dataset directory if the metadata
write fails, avoiding half-created library records. SAM3/Qwen dataset conversion
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

Data ingestion cancellation must stop finalization before creating new result
artifacts. If cancellation is observed after candidate/reference encoding but
before the analysis result write, no `result.json` or embeddings cache is
published. If cancellation is observed after local reference-profile training but
before the head write, no local SALAD head is added.
