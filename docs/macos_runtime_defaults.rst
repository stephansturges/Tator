macOS runtime defaults
======================

This supplements the `macOS inference setup <macos_inference_setup.md>`_.
The defaults below apply when starting Tator with
``tools/run_macos_backend.sh`` and the macOS inference requirements profile.
The launcher reads ``.env.macos`` before applying its defaults.

CLIP loads when requested
------------------------

The launcher defaults ``TATOR_SKIP_CLIP_LOAD`` to ``1``. This avoids loading
the optional CLIP backbone during backend startup. It does not remove CLIP
support: classifier activation can load the selected model, and feature
extraction for a classifier head can load its configured backbone when needed.

A saved active classifier may therefore initially report that its encoder is
not ready. Activate the classifier before using workflows that require an
already-loaded encoder; not every legacy endpoint performs lazy activation.
To request startup loading instead, set ``TATOR_SKIP_CLIP_LOAD=0`` in
``.env.macos``. Startup loading still requires a configured active CLIP
classifier; the setting does not select or create one.

Classifier compatibility
------------------------

``requirements-macos-inference.txt`` pins scikit-learn to ``1.7.2`` because
the default classifier artifact was serialized with that version. This keeps
the runtime and saved classifier compatible instead of suppressing a version
mismatch warning. The macOS setup continues to use Python 3.11.

The pin is specific to this profile and artifact. It does not establish
compatibility with arbitrary classifier pickles created using other versions.
The full Linux/GPU requirements profile is unchanged.

Startup messages and overrides
------------------------------

The launcher defaults ``NO_ALBUMENTATIONS_UPDATE`` to ``1`` to disable its
import-time update notice. Albumentations 2.x is outside this profile's
supported dependency range. This setting does not disable augmentations or
change the installed package.

Three known upstream import warnings are filtered: the ``pkg_resources``
deprecation in ``sam3.model_builder``, the non-CUDA autocast warning in
``sam3.model.sam3_video_inference``, and the deprecated compatibility import
in ``timm.models.layers``. Other warnings and runtime errors remain visible.

Explicit environment settings override the launcher defaults. Caller-provided
``PYTHONWARNINGS`` filters are applied after the three defaults, so a stricter
warning policy can override them. Configure persistent overrides in the
environment file selected by ``ENV_FILE`` (``.env.macos`` by default).

Local working files
-------------------

``.agent-state/`` is ignored by Git because it holds local agent execution
notes and state. It is not a model, dataset, or runtime configuration folder
and is not required to run Tator.
