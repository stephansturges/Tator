Hosted saved SAM3 recipes
========================

``services.hosted.sam3_chain.Sam3ChainRuntime`` executes an administrator's
installed recipe package in the user's sequential private worker. The request
selects attached recipe IDs and explicit threshold, class and cascade settings.
It cannot supply recipe files, classifier paths or inline recipe bodies.
Results are proposals for the annotation canvas; this adapter writes no labels.

The installed package retains the original recipe JSON, positive and negative
crops, and optional ``clip_head/head.npz`` and ``clip_head/meta.json``. Its exact
file inventory and every digest are verified before execution. SAM3 and optional
extra classifier component IDs resolve to pinned installed definitions. Embedded
NumPy/pickle heads are executable trusted administrator input; they must never
be installed from an annotator's request. Class bindings resolve recipe labels
to the current stable dataset class IDs and numeric positions.

Execution contract
------------------

The adapter recognizes the original ``sam3_steps``, ``sam3_greedy`` and legacy
prompt/exemplar schemas. It preserves explicit step order, enabled steps, text
seed thresholds, seed deduplication, trained-head seed gates, bounded visual
expansion, final trained-head gates and optional extra classifier gates. Saved
crop views, fitted feature transforms, calibrated head probabilities and
background guards use the existing classifier mechanisms. A missing required
head fails; it does not silently turn into unfiltered SAM inference.

Retained crop banks use SAM3's native geometry encoder. Each retained crop is
encoded with its positive or negative box label. Those visual prompt tokens
enter the private network's ``_encode_prompt`` external-visual-input seam while
the decoder processes the unchanged target image. The temporary instance
binding is restored in ``finally``. The bank runs once at the first recipe
stage, or for the explicit exemplar of a legacy stage; subsequent text stages
do not replay that global bank. ``use_negative_exemplars=False`` and explicit
``negative_strength=0`` disable the negative bank.
Recipes with only negative crops use their saved text target and the native
negative visual tokens together on each text stage. They do not require an
invented positive crop.

Cascade suppression preserves per-class IoU, optional cross-class suppression,
groups and participation opt-outs. Ranking can use SAM confidence or the saved
head's class probability/margin. Classifier ranking does not overwrite the
returned original SAM score. Evidence includes admitted recipe digests, stage
inputs and outputs, retained crop digests, full classifier probabilities and
gate decisions, ranking values, and stage/cascade counts. No server paths or
credentials are included.

``clip_head_background_penalty`` belongs to the recipe-mining optimizer, as
labelled by the existing control. It is validated and retained as optimizer
provenance; applying an already selected recipe does not subtract it from SAM
confidence or classifier probabilities.

This is a functional repair of the checked-in legacy per-recipe executor. That
executor is a prompt-loop stub and does not implement the saved classifier,
expansion or crop-bank stages; its mode comparison also differs from the schema
classifier's returned values. The hosted implementation therefore does not claim
exact numerical parity with a previously complete legacy executor.

Explicit limits and qualification
---------------------------------

An active negative bank with a weighted legacy ``negative_strength`` other than
zero or one is rejected. The retained source and Git history contain only its
serialization and generated zero value; there is no executable weighted-negative
equation. An inactive negative bank does not block execution. Native SAM3
positive/negative geometry labels are binary. Multiplying their embeddings or
subtracting a classifier score would invent a different recipe.

The native classifier training and inference paths treat the saved
``sam3_mask_fusion_v1`` aggregation value as ordinary encoder pooling. The hosted
classifier preserves that actual feature path, validates the resulting feature
dimension against the retained head and records the declared aggregation. The
separate DQE mask descriptor path is not part of this classifier trainer.
``local_salad`` remains explicitly disabled by the original Auto Class contract.
``sam3_mask_salad_fusion_v1`` requires the exact trained SALAD runtime artifact:
the original inference path requires ``_class_analysis_salad_head``, while the
ordinary saved classifier loader retains only metadata and no such runtime.
Reconstructing that descriptor requires registering the original SALAD weights
and their binding to this classifier. Selecting or training another SALAD head
would not preserve the trained feature space. That artifact binding remains a
compatibility gap.

Requests allow at most 32 cascade entries, 64 recipe stages, 32 prompts per
stage, 500 visual seeds per stage, 64 retained crop-bank entries, 10,000 recipe
candidates and 16 MiB of stage evidence. A crop is limited to 16 MiB and
16 million pixels. Existing prompt-runtime mask and candidate bounds also
apply. Exceeding a bound raises an explicit error instead of silently dropping
stages or evidence.

Run the focused mechanism checks with::

    .venv-macos/bin/python -m pytest -q tests/test_hosted_sam3_chain.py

They load actual retained NPZ heads and run production crops, feature composition,
classifier probability/gate math and cascade logic. CPU model doubles isolate
the SAM3/encoder boundary. They verify stage order, rejected seeds, expansion,
final filtering, positive/negative external-token plumbing, failure cleanup,
classifier ranking, class/group identity and artifact rejection. They do not
qualify actual SAM3 or encoder weights, crop-bank usefulness, GPU memory, target
checkpoint compatibility, or three-user performance. Those require the user's
GPU machine and actual registered packages before production acceptance.
