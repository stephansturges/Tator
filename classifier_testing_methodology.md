# Classifier Testing Methodology

This note documents how we evaluate classifier heads (CLIP / DINOv3) so results
are comparable across runs and easy to extend.

## Data + Embeddings
- Dataset: YOLO images + labels (e.g. `uploads/qwen_runs/datasets/qwen_dataset` +
  `uploads/clip_dataset_uploads/qwen_dataset_yolo/labels`).
- Embeddings are cached under `uploads/clip_embeddings/<signature>`.
- The signature is derived from:
  - encoder type + model name
  - labelmap content + hash
  - background class count + sampling policy
  - augmentation policy + oversample policy
  - image + label file mtimes

## Split Strategy
- We use **GroupShuffleSplit** by image (group = image path) to prevent
  per-image leakage.
- Default split: `test_size=0.2`, `random_seed=42` (from training meta).
- If a group split is not possible, fall back to a stratified split by class.

## Metrics
We report:
- `accuracy` (overall)
- `macro_f1_all` / `weighted_f1_all` (all classes)
- `macro_f1_fg` / `weighted_f1_fg` (foreground only)
- `macro_precision_fg` / `macro_recall_fg` (foreground only)

Foreground-only metrics exclude any `__bg_*` classes from the CLIP/DINOv3
negative-class augmentation.

## Where Results Live
We append/update three files:
- `clip_dinov3_metrics_20241224.json`
- `clip_dinov3_metrics_20241224.csv`
- `clip_dinov3_metrics_20241224.txt`

Each entry includes: encoder type/model, embedding dim, metrics (including
foreground precision/recall when available), convergence
info, and artifact paths.

## How To Extend (New Run)
1. Train a new head with `tools/train_clip_regression_from_YOLO.py`.
2. Reuse cached embeddings (`--reuse-embeddings`) to keep splits consistent.
3. Evaluate on the **same split** (grouped by image) used in training.
4. Compute metrics from predicted labels:
   - All classes
   - Foreground-only (exclude `__bg_*`)
5. Append a new row to the JSON/CSV/TXT tables.

## Notes
- MLP heads store explicit layer activations (`relu` for hidden, `linear` for
  output). Evaluation should respect these when reconstructing inference.
- If the evaluation split size does not match training metadata, treat it as a
  mismatch and re-check split parameters.
