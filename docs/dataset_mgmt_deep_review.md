# Dataset Management Deep Review (Linked/Transient/Annotation Paths)

Date: 2026-03-02 (UTC)

## Scope
This review covers all newly added dataset-management paths and their UI wiring:

- Backend routes/contracts:
  - `api/datasets.py`
  - `localinferenceapi.py`
  - `services/datasets.py`
- Frontend behavior and UX copy:
  - `ybat-master/ybat.html`
  - `ybat-master/ybat.js`
- Current automated coverage:
  - `tests/test_dataset_linked_annotation_flows.py`
  - `tests/test_dataset_download_cleanup.py`

Review method:
- End-to-end flow trace (UI -> API -> state/storage -> list/readback)
- Threat-model pass (path safety, lock safety, data leakage)
- Concurrency/integrity pass
- UI explanation clarity pass (operator-facing, minimal)

---

## E2E Flow Status

1. Register server path (`/datasets/register_path`): **Functional with caveats**
- Works and persists linked dataset registry metadata.
- Caveat: no duplicate suppression by `linked_root` or signature.

2. Open server path transient (`/datasets/open_path`): **Functional with caveats**
- Works and creates in-memory transient session.
- Caveat: session lifecycle is unbounded (no TTL purge / explicit destroy endpoint).

3. Save transient -> library (`/datasets/transient/{session_id}/save`): **Functional**
- Works and persists overlay labels/text into `.annotation_overlay` under registry root.

4. Persistent annotation lifecycle (`/datasets/{id}/annotation/*`): **Integrity issues found**
- Session lock endpoints exist.
- Snapshot/meta write endpoints do not enforce lock ownership.

5. Transient annotation lifecycle (`/datasets/transient/{sid}/annotation/*`): **Functional with caveats**
- Read/write and lock endpoints exist.
- Same lock-enforcement gap on write semantics (conceptually, since writes are in-memory and callable without lock check).

6. Linked delete/download semantics: **Mostly correct**
- Linked delete removes only registry record (source not deleted).
- Download overlays correctly replace effective labels/text.

7. Web UI path controls and status: **Functional, under-explained**
- New controls are wired.
- Important semantics are not explicit enough in UX copy (what is persisted, what is transient, what delete means for linked).

---

## Ranked Findings

## Critical

### C1) Annotation writes do not enforce lock ownership
- Files:
  - `localinferenceapi.py:10341`
  - `localinferenceapi.py:10395`
- What happens:
  - `save_dataset_annotation_snapshot()` and `patch_dataset_annotation_meta()` write data regardless of current lock holder/session.
- Impact:
  - Concurrent editors can overwrite each other even when lock API reports a lock.
  - Lock is advisory, not protective.
- Evidence:
  - Direct call to `save_dataset_annotation_snapshot()` succeeds without a session/lock and writes overlay files.
- Fix:
  - Add a shared lock guard (`_require_annotation_lock_owner(...)`) and call it from all mutating annotation endpoints (persistent and transient variants).
  - Return `409` (`annotation_lock_active`) for lock mismatch.

### C2) Lock can be released without session id
- File:
  - `localinferenceapi.py:9633`
- What happens:
  - `_annotation_release_lock()` clears lock when `session_id` is omitted.
- Impact:
  - Any caller can clear active lock by calling stop endpoint with empty payload.
- Evidence:
  - `_annotation_release_lock(meta_with_active_lock, {})` returns `ok=True` and empties lock.
- Fix:
  - Require matching `session_id` for unlock unless explicit `force=true`.
  - Add negative tests for sessionless stop and wrong-session stop.

## High

### H1) Annotation manifest leaks server filesystem path
- File:
  - `localinferenceapi.py:9594`
- What happens:
  - `get_dataset_annotation_manifest()` returns `meta_path` (absolute server path) to clients.
- Impact:
  - Unnecessary host path disclosure.
- Fix:
  - Remove `meta_path` from API response; keep internal only.

### H2) Transient sessions have no TTL enforcement / cleanup
- Files:
  - `localinferenceapi.py:8748`
  - `localinferenceapi.py:10224`
- What happens:
  - `DATASET_TRANSIENT_SESSIONS` grows indefinitely; no expiration/purge.
- Impact:
  - Memory growth and stale session ambiguity.
- Fix:
  - Add `transient_created_at` + `transient_expires_at`, enforce on resolve, and periodic purge.
  - Add explicit `DELETE /datasets/transient/{session_id}` endpoint.

### H3) Text label pathing collapses nested image paths by stem
- Files:
  - `localinferenceapi.py:9356`
  - `localinferenceapi.py:9360`
- What happens:
  - Text labels are keyed as `text_labels/<stem>.txt`.
  - `subdir1/a.jpg` and `subdir2/a.png` collide into `a.txt`.
- Impact:
  - Silent cross-image overwrite risk.
- Fix:
  - Use relative image path with suffix stripped (e.g., `text_labels/<relpath>.txt`), not stem-only.
  - Provide migration adapter for legacy flat `text_labels/*.txt`.

## Medium

### M1) Linked path registration is duplicate-prone
- File:
  - `localinferenceapi.py:10162`
- What happens:
  - Same linked path can be repeatedly registered as new IDs.
- Impact:
  - Library clutter and user confusion.
- Fix:
  - On register, detect existing registry record by `linked_root` and/or source signature; return existing unless `force_new`.

### M2) Open/register path does not strongly validate dataset shape
- Files:
  - `localinferenceapi.py:10188`
  - `localinferenceapi.py:10162`
- What happens:
  - Any allowlisted directory can be opened/registered.
- Impact:
  - Unknown/non-YOLO paths appear as datasets; poor operator feedback.
- Fix:
  - Add strict mode validation (`labelmap + image/label dirs`) with clear error details.

### M3) Broken linked roots silently degrade to registry-root view
- File:
  - `services/datasets.py:307`
- What happens:
  - If `linked_root` is missing/bad, entry remains but `dataset_root` effectively points to registry dir.
- Impact:
  - Confusing behavior and stale records.
- Fix:
  - Add `linked_root_status` (`ok|missing|invalid`) and surface warning in API/UI.

### M4) UI wording does not fully explain persistence semantics
- Files:
  - `ybat-master/ybat.html:3184`
  - `ybat-master/ybat.js:2750`
- What happens:
  - Current copy does not fully explain transient persistence boundaries and linked delete behavior.
- Impact:
  - Operator confusion and accidental misuse.
- Fix:
  - Add explicit helper copy/tooltips (see UI section below).

---

## Threat-Model Notes

Passes:
- Absolute path + allowlist root guard for server-path opening/registration (`_validate_linked_dataset_path`).
- Linked delete protection now prevents deleting external source files.
- Overlay export behavior correctly applies effective overlay files.

Open risks:
- Lock model is currently bypassable on write paths (Critical).
- Host path leakage via manifest response (High).
- Transient session lifecycle is unbounded (High).

---

## UI Explanation Recommendations (Operator-Clear Minimal)

Recommended replacement/help text:

1. For server-path section title help (`ybat-master/ybat.html:3184`):
- Current intent is good; add explicit persistence semantics:
- Suggested copy:
  - "Open transient: temporary backend session (lost on backend restart)."
  - "Save transient to library: persists metadata + overlay labels/text, does not copy source images."
  - "Register path in library: adds persistent linked dataset record only."

2. For linked dataset badge tooltip (`ybat-master/ybat.js:2784`):
- Add explicit delete behavior:
  - "Delete removes only this library record and overlay metadata; source files remain on disk."

3. For path summary (`ybat-master/ybat.js:2703`):
- Include transient warning:
  - "Transient session is temporary until saved to library."

4. For save button tooltip/state:
- Add disabled reason when no transient session is open.

---

## Test Coverage Gaps

Current tests cover:
- Linked delete safety
- Transient save overlay persistence
- Linked download overlay precedence
- Linked glossary write location

Missing high-priority tests:
1. Lock enforcement:
- snapshot/meta writes fail with `409` when no session lock or wrong session.
2. Unlock rules:
- stop without session id fails unless `force=true`.
3. Manifest contract:
- assert `meta_path` is not present.
4. Text-label collisions:
- nested duplicate stem paths do not collide after keying fix.
5. Transient session expiry:
- expired session access returns deterministic `404`/`410` and is purged.

---

## Implementation-Ready Remediation Plan

1. **Lock correctness hardening**
- Add `_require_annotation_lock_owner(meta, payload)` helper.
- Enforce in:
  - `save_dataset_annotation_snapshot`
  - `patch_dataset_annotation_meta`
  - `save_transient_annotation_snapshot`
  - `patch_transient_annotation_meta`
- Tighten `_annotation_release_lock` to require session id unless `force=true`.

2. **Response hygiene**
- Remove `meta_path` from manifest response.

3. **Transient lifecycle management**
- Add session expiry + purge and explicit destroy endpoint.
- UI: show transient TTL/expiry state.

4. **Text-label keying migration**
- Switch to relpath-based text label storage.
- Read-path fallback for legacy stem-only files.
- Add migration tool or lazy-migrate on write.

5. **Linked dataset dedupe + health status**
- Dedupe by `linked_root`/signature at registration.
- Expose `linked_root_status` and warn in UI for broken links.

6. **UI copy update pass**
- Apply minimal explicit wording above.
- Add tooltip for linked delete semantics and transient persistence.

---

## Acceptance Criteria for Fix Phase

- Lock mismatch blocks all annotation write endpoints with explicit `409` details.
- Sessionless unlock no longer works.
- Manifest no longer exposes host metadata paths.
- Transient sessions expire and are cleaned up.
- Text label storage is collision-safe for nested paths.
- UI clearly communicates transient vs persistent behavior and linked delete semantics.
