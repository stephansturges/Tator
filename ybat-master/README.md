# Tator Browser UI

`ybat-master/` contains the static browser workspace used by Tator. It began as
Ybat, but this copy is now the Tator labeling interface: dataset-aware
annotation, assisted class prediction, SAM/SAM3 tools, Qwen captioning, detector
prepasses, EDR application, and export controls all live here.

Start the backend from the repo root:

```bash
tools/run_macos_backend.sh
```

Then open the UI served by the backend:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/tator.html
```

The UI talks to the backend configured by `API_ROOT`, which defaults to
`http://localhost:8000`. The old `/ybat.html` URL redirects to `/tator.html`.

For frontend-only development, you can also serve the directory with a small
static server. This does not start backend APIs; keep the backend running on
`http://localhost:8000` when using live features:

```bash
python3 -m http.server 8080 -d ybat-master
```

Then open:

```text
http://127.0.0.1:8080/tator.html
```

From another directory, `cd` to your clone first before running backend or
static-server commands.

## Main Areas

- **Label Images**: manual box/polygon labeling, class cycling, full-screen
  image mode, SAM/SAM3 prompts, detector suggestions, Qwen captions, and export.
- **Dataset Management**: upload/register datasets, inspect linked-path health,
  edit labelmaps and glossaries, and open datasets for labeling.
- **Training**: CLIP/DINO class predictors, YOLO, RF-DETR, SAM3, and Qwen job
  controls.
- **Backend Config**: runtime status, predictor slots, Qwen runtime settings,
  and install/system checks.
- **EDR and Prepass**: build, calibrate, save, load, and apply reusable
  prelabeling recipes.

## Development Notes

- `ybat.js` is intentionally plain browser JavaScript. Keep additions local to
  the existing helper/state pattern and run `node --check ybat-master/ybat.js`
  after edits.
- Bump the `ybat.js?v=...` cache key in `tator.html` whenever frontend behavior
  changes.
- Keep top-tab navigation on the early delegated click handler; individual
  panel initialization must not be able to leave visible tabs inert.
- Imported YOLO/VOC/COCO labels are stamped with UUID and creation metadata so
  SAM and auto-tweak responses can target the correct box.
- Zip imports skip directory entries and keep importing remaining files if one
  archived label file fails.
- The frontend should remain usable from local file mode, but a static server is
  better for development because it avoids stale browser cache and path issues.

## Validation

Fast checks for frontend-only edits:

```bash
node --check ybat-master/ybat.js
git diff --check
```

For changes that touch backend endpoints used by the UI, also run the relevant
pytest slice and verify the backend:

```bash
.venv-macos/bin/python -m pytest tests/test_api_route_uniqueness.py -q
curl http://127.0.0.1:8000/system/health_summary
```

The root [readme.md](../readme.md) has the full repository map and setup paths.
