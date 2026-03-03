# Dataset Annotation Feature Acceptance Checklist

Use this checklist after deploying frontend changes that enable loading server datasets directly into the annotation tab.

## Automated Playwright Coverage
`tests/ui/e2e/test_dataset_annotation_flows.py` now covers the critical dataset-annotation UX path end-to-end:
- Dataset Manager default messaging/disabled-state clarity checks.
- Transient open -> annotation workspace load.
- Register path workflow + linked dataset card open-in-annotation flow.
- Read-only lock takeover flow (external lock holder -> takeover -> writable).
- Save/close safety when snapshot save fails (close blocked, then recover).
- Stale async manifest response safety (close during in-flight load does not resurrect state).

Run command:

```bash
RUN_UI_E2E=1 \
UI_PAGE_URL=http://127.0.0.1:8001/ybat.html \
UI_DATASET_PATH=/abs/path/to/yolo_dataset \
UI_API_ROOT=http://127.0.0.1:8000 \
UI_HEALTH_URL=http://127.0.0.1:8000/system/health_summary \
pytest -q tests/ui/e2e/test_dataset_annotation_flows.py -m ui_smoke
```

Notes:
- Backend (`uvicorn`) and static UI host must both be running before this command.
- By default these tests are skipped unless `RUN_UI_E2E=1` and required env vars are set.
- Full matrix run: `pytest -q tests/ui/e2e -m \"ui_full\"`
- PR smoke (functional + usability): `pytest -q tests/ui/e2e -m \"ui_smoke or ui_usability_smoke\"`
- Nightly full (functional + usability): `pytest -q tests/ui/e2e -m \"ui_full or ui_usability_full\"`
- Control claim validation: `python tools/check_playwright_control_coverage.py`

## Preconditions
- Backend is running and reachable.
- At least one valid dataset path exists on server disk (`train/images`, labels, labelmap).
- UI is loaded from the same environment as the backend.

## 1) Transient open into annotation
- Go to `Dataset Management`.
- Enter a valid server path in `Server path`.
- Click `Open transient`.
- Click `Open in annotation`.
- Expected:
  - Label Images tab opens.
  - `Annotation source` shows `transient dataset`.
  - `imageList` contains images.
  - Local file selectors are disabled.

## 2) Linked dataset open into annotation
- In `Dataset Management`, use a dataset card action `Open in annotation`.
- Expected:
  - Label Images tab opens.
  - `Annotation source` shows `linked dataset`.
  - Annotation panel shows dataset id/label and progress.

## 3) Failed open preserves current workspace
- First load local images/classes in Label Images.
- Trigger an annotation open failure (invalid session/id/path or backend failure).
- Expected:
  - Error message is shown.
  - Existing local images/classes remain loaded (no workspace wipe).
  - Annotation source returns to `local files`.

## 4) Read-only lock behavior
- Hold lock from another editor/session.
- Open same dataset in this UI.
- Expected:
  - Annotation source shows read-only with lock holder text.
  - BBox create/delete is blocked.
  - Text/caption input is disabled.
  - `Take over lock` is enabled.
- Click `Take over lock`.
- Expected:
  - Read-only message clears.
  - Edit controls re-enable.

## 5) Save + close safety
- Make edits that create dirty state.
- Force snapshot save failure (network drop or backend error).
- Click `Close dataset`.
- Expected:
  - Close is blocked.
  - Error indicates save failed and dataset stays open.
- Recover backend and click `Save now`.
- Click `Close dataset`.
- Expected:
  - Session closes cleanly.
  - Annotation source switches to `local files`.

## 6) Stale request safety
- Start opening dataset into annotation and immediately click `Close dataset`.
- Expected:
  - UI remains in `local files` mode.
  - Delayed manifest response does not repopulate annotation session afterward.
