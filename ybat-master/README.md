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

The root [readme.md](../readme.md) describes installation and the complete
product workflow.
