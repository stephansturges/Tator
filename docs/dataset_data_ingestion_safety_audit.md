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
| Annotation snapshot/meta save | Writes backend overlay labels/text and metadata | Active-lock ownership, image existence checks, guarded overlay roots, atomic no-follow writes | `tests/test_dataset_linked_annotation_flows.py` |
| Dataset glossary save | Writes canonical glossary into backend metadata | Guarded metadata root and strict metadata write | `tests/test_dataset_linked_annotation_flows.py`, `tests/test_glossary_library.py` |

## Data Ingestion

| Flow | Data touched | Guard | Coverage |
|---|---|---|---|
| Build reference profile | Creates job staging data and a local reference-profile head | Guarded job root, upload size/quota checks, backend-reference source validation, strict local-head path checks | `tests/test_data_ingestion.py` |
| Analyze candidates | Creates job staging data, result JSON, embeddings cache | Guarded job root, candidate/reference gating, matching reference-profile validation, guarded result reads | `tests/test_data_ingestion.py`, UI E2E |
| Backend reference dataset use | Reads existing dataset images | Dataset id resolution, path containment inside selected dataset root, active-job delete blocking by dataset id | `tests/test_data_ingestion.py`, `tests/test_dataset_linked_annotation_flows.py` |
| Cancel job | Mutates in-memory job state | Terminal jobs are not cancellable; active jobs move through `cancelling` | `tests/test_data_ingestion.py` and endpoint sanity |

## Current Invariant

Dataset Management should not report success for a library mutation unless the
durable metadata record was actually written. Upload, register, and transient
save paths now roll back their backend-created dataset directory if the metadata
write fails, avoiding half-created library records.
