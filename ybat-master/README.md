# Tator browser UI

This directory contains the shared annotation canvas and the hosted dataset
workspace. The canvas began as Ybat and retains its drawing tools and keyboard
shortcuts, with model-assisted annotation added by Tator.

## Hosted workspace

The hosted service serves `hosted.html` at `/`: sign in, select a dataset and
open an image from the work queue. It serves the permitted annotation canvas at
`/annotate?image=<image-id>` using `tator.html` with hosted integration scripts.

- `hosted.js` and the `hosted_*_ui.js` modules implement dataset management,
  assignments, review, model settings, analysis and operations.
- `hosted_canvas.js` connects the shared canvas to authenticated media, versioned
  drafts, editor leases, submission and recovery.
- `hosted.css` styles the workspace; `hosted_canvas.css` adapts the annotation UI.
- `ybat.js` contains the shared annotation tools. Hosted adapters route assistance
  through the current user's authorized image and configured server models.

Hosted requests use the same origin and the signed-in session. Opening the HTML
files directly or pointing a static file server at this directory does not start
that workflow. Follow [hosted deployment](../docs/hosted_deployment.rst) and the
[administrator/annotator guide](../docs/hosted_workflow.rst).

## Local workbench

From the repository root, start the existing macOS backend:

```bash
tools/run_macos_backend.sh
```

Open `http://127.0.0.1:8000/` or `/tator.html`; `/ybat.html` redirects to the latter.
This mode retains the local dataset library, ingestion, quality exploration,
training and backend configuration panels. It has no hosted assignment or
approval workflow.

The local UI uses its configured `API_ROOT`, defaulting to the page's HTTP(S)
origin unless an override was saved. Outside HTTP(S), the fallback is
`http://localhost:8000`. Hosted mode uses authenticated same-origin adapters; it
must not be connected to the unauthenticated local inference API as a deployment
shortcut.

See the [project overview](../readme.md), [documentation index](../docs/index.rst)
and [local environment setup](../docs/environment_setup.md).
