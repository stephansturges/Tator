# Playwright UI Usability Coverage Checklist

This checklist documents the usability-focused Playwright coverage added under `tests/ui/e2e/`.

## Scope
- Popup/dialog readability and visibility.
- Tooltip readability.
- Status/toast legibility.
- Horizontal overflow checks on visible tab panels.
- Targeted WCAG-AA-oriented checks on critical text contrast and minimum font size.

## Test Modules
- `tests/ui/e2e/test_usability_popups_contract.py`
- `tests/ui/e2e/test_usability_tooltips_toasts_contract.py`
- `tests/ui/e2e/test_usability_legibility_contract.py`

## Case IDs
- `UX_MODAL_BACKGROUND_LOAD_READABLE`
- `UX_MODAL_BATCH_TWEAK_READABLE`
- `UX_TOOLTIP_READABILITY_PREPASS`
- `UX_STATUS_NOTICE_LEGIBILITY`
- `UX_TAB_PANELS_NO_OVERFLOW`
- `UX_CRITICAL_TEXT_CONTRAST`

## Run Commands

PR-oriented smoke:

```bash
RUN_UI_E2E=1 \
UI_PAGE_URL=http://127.0.0.1:8001/ybat-master/ybat.html \
UI_DATASET_PATH=/abs/path/to/yolo_dataset \
UI_API_ROOT=http://127.0.0.1:8000 \
UI_HEALTH_URL=http://127.0.0.1:8000/system/health_summary \
pytest -q tests/ui/e2e -m "ui_smoke or ui_usability_smoke"
```

Nightly/full:

```bash
RUN_UI_E2E=1 \
UI_PAGE_URL=http://127.0.0.1:8001/ybat-master/ybat.html \
UI_DATASET_PATH=/abs/path/to/yolo_dataset \
UI_API_ROOT=http://127.0.0.1:8000 \
UI_HEALTH_URL=http://127.0.0.1:8000/system/health_summary \
pytest -q tests/ui/e2e -m "ui_full or ui_usability_full"
```

Coverage manifest contract:

```bash
python tools/check_playwright_control_coverage.py
```

Optional artifact output path:

```bash
export UI_E2E_ARTIFACT_DIR=tmp/ui_e2e/usability
```
