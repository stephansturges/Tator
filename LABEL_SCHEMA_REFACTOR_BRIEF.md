# Label Schema / Label Ordering Refactor Brief (Future Work)

## Problem statement (what keeps breaking)

We currently have **multiple “sources of truth” for labels**:

- YOLO: numeric class ids are **0-based** and implicitly mean “index into `labelmap.txt` / label list”.
- COCO: `category_id` is an **arbitrary integer** (often 1-based and contiguous, but not guaranteed).
- Qwen JSONL: labels are **strings** (`{"label": "light_vehicle", ...}`).
- CLIP LogReg heads: `clf.classes_` ordering is **sklearn-defined** (sorted), not necessarily dataset/labelmap order.
- UI: frequently shows “class id” and “class name” without making clear which id space we’re talking about.

This causes:

- Confusion (“why is `light_vehicle` index 7?”) even when technically correct (sklearn ordering).
- Real bugs when any code path assumes:
  - `category_id == idx + 1`, or
  - “label list index == classifier proba index”, or
  - category ids are contiguous and start at 1.
- Brittle backwards compatibility when datasets are converted, re-split, used for training, then mined again (the label order can drift if any step reorders categories).

We want **stable, portable, explicit label semantics** across the repo.

## Desired end state (goals)

1. **One canonical label schema** used everywhere (dataset conversion, CLIP/Qwen/SAM3 training, recipe mining, UI).
2. Default to **name-based mapping** whenever possible, but always preserve a **stable canonical order** for:
   - writing COCO categories,
   - writing YOLO `labelmap.txt`,
   - CLIP head probability vector interpretation.
3. Every artifact that can be reused later includes enough metadata to be **self-describing**:
   - dataset: schema stored next to dataset meta,
   - trained models (CLIP/Qwen/SAM3): schema hash + ordered class list,
   - recipes: schema hash + ordered class list (already partially done via `labelmap_hash` + `labelmap`).
4. Validation at boundaries (dataset load, conversion, model activation, mining start):
   - warn loudly or fail fast on schema mismatches,
   - but provide a controlled “force / remap / reconcile” path when users know what they’re doing.

Non-goals (for the first iteration):

- Perfect auto-merging of label schemas across unrelated datasets.
- Renaming or splitting/merging classes automatically.
- Changing the user-facing class names everywhere (we should preserve existing labels).

## Terminology (avoid ambiguous “class id”)

We should be explicit in code/docs/logs:

- **`schema_index`**: 0-based index into the canonical ordered class list.
- **`schema_class_id`**: stable integer id for a class in our canonical schema (may be 1..N, but we should not assume).
- **`yolo_class_id`**: numeric id stored in YOLO `.txt` labels (0-based by definition).
- **`coco_category_id`**: numeric id in COCO annotations (`category_id`) (arbitrary integer).
- **`class_name`**: canonical string label (stable key when available).
- **`alias`**: alternate spellings/synonyms used only for matching (never emitted as the canonical name unless explicitly chosen).

Rule of thumb:
- Use **names** to match (best-effort).
- Use **schema ids/indexes** to serialize and compute (deterministic).

## Proposed canonical representation: `LabelSchema`

### Data model (in-memory)

Create a small utility module (new file) to hold and validate the schema:

- `LabelClass`
  - `schema_class_id: int` (stable id)
  - `name: str` (canonical)
  - `aliases: list[str]` (optional; for matching only)
  - optional: `display_name`, `color`, `supercategory`, etc (future)

- `LabelSchema`
  - `version: int`
  - `classes: list[LabelClass]` in canonical order
  - derived maps:
    - `id_to_index`, `index_to_id`
    - `name_to_id` (normalized)
    - `id_to_name`
  - methods:
    - `normalize_name(s: str) -> str` (lowercase + whitespace/underscore/hyphen normalization)
    - `match_name_to_id(name: str) -> Optional[int]` (name + aliases)
    - `hash() -> str` (stable hash of `(id, canonical_name)` in canonical order)
    - `validate_unique()` (no duplicate ids, no duplicate normalized names/aliases)

### On-disk format (JSON)

Add a new file next to dataset metadata:

- `label_schema.json` (preferred)
  - allows both Qwen datasets (`dataset_meta.json`) and SAM3 datasets (`sam3_dataset.json`) to share it.

Suggested minimal JSON:

```json
{
  "version": 1,
  "hash": "0123abcd4567",
  "classes": [
    { "id": 1, "name": "light_vehicle", "aliases": ["car", "automobile"] },
    { "id": 2, "name": "person", "aliases": [] }
  ]
}
```

Notes:
- `hash` is redundant (can be computed) but convenient for quick mismatch checks.
- Keep `aliases` optional; do not require synonyms everywhere.
- Do not store computed maps in JSON (derive them in code).

### Canonical id strategy (decision needed)

Two reasonable options:

**Option A (keep current behavior):** canonical ids are contiguous 1..N in the same order as labelmap.
- Pros: matches current COCO conversions (we already write `category_id = class_idx + 1`).
- Cons: still confusing for users who think in YOLO ids (0..N-1), but that can be solved by showing both.

**Option B:** canonical ids equal YOLO ids (0..N-1), and COCO uses those ids directly.
- Pros: fewer “+1/-1” conversions.
- Cons: some COCO tooling expects 1-based ids (not strictly required, but common).

Recommendation: **Option A** to minimize churn, but make sure we never assume contiguity/starting point.

## Migration plan (incremental, backwards compatible)

### Phase 0 — Observability + guardrails (low risk)

- [ ] Add a single “schema summary” log line whenever:
  - dataset is loaded for mining,
  - dataset is converted to COCO,
  - CLIP model is activated,
  - mining begins for a class.
- [ ] In logs/UI, always show:
  - `class_name`
  - `coco_category_id` (or schema id if aligned)
  - `schema_index` (if known)
  - if CLIP head attached: `clip_head_index`

### Phase 1 — Introduce `LabelSchema` utilities (core refactor enabler)

- [ ] Add `label_schema.py` (or similar) with:
  - load/save JSON
  - schema inference from:
    - `label_schema.json`
    - `sam3_dataset.json` / `dataset_meta.json`
    - COCO `categories`
    - YOLO `labelmap.txt`
  - validation + stable hash
- [ ] Add a single “schema loader” function used everywhere:
  - `load_label_schema_for_dataset(dataset_root) -> LabelSchema`

### Phase 2 — Dataset ingestion + conversion correctness

- [ ] On dataset upload (`/datasets` zip YOLO import):
  - write `label_schema.json` derived from `labelmap.txt`
  - ensure `sam3_dataset.json` includes `label_schema_hash` (or embeds schema)
- [ ] On Qwen dataset finalize:
  - write `label_schema.json` derived from `dataset_meta.json["classes"]`
  - validate labels in JSONL belong to schema (at least warn)
- [ ] On COCO conversion:
  - stop building categories from ad-hoc rules; instead:
    - derive schema first,
    - write COCO `categories` strictly from schema in canonical order,
    - map each annotation’s label to the correct `coco_category_id`.

### Phase 3 — Training pipelines (CLIP/Qwen/SAM3)

#### CLIP training/export

Goal: a trained CLIP head must be **unambiguous** about its output ordering.

- [ ] Decide on a single rule:
  - Either train with numeric y labels (schema_index or schema_class_id) so `classes_` order is deterministic,
  - OR reorder the fitted model at export time so `classes_` matches schema order.
- [ ] Export a `clip_head_meta.json` (or extend existing meta) that includes:
  - `label_schema_hash`
  - `classes_in_schema_order` (canonical names list)
  - `classes_in_model_order` (what `clf.classes_` actually is)
  - an explicit `model_index_by_schema_index` mapping (or inverse)
- [ ] During activation (`/clip/active_model`):
  - validate compatibility between provided labelmap + classifier + dataset schema (when a dataset is selected).
  - warn if `clf.classes_` is not aligned to schema and we’re relying on name matching.

#### Qwen training

- [ ] Ensure Qwen dataset meta includes schema.
- [ ] When building training config:
  - carry `label_schema_hash` into run metadata,
  - optionally embed “allowed classes” in the prompt (if helpful), but do not reorder labels in the JSONL.
- [ ] When converting Qwen JSONL -> COCO:
  - map by canonical name -> schema id.

#### SAM3 training

- [ ] Replace all uses of `cat_id = idx + 1` with schema-driven ids.
- [ ] When supplying prompt variants to the COCO loader:
  - allow user prompts keyed by any of:
    - canonical name
    - schema id
    - schema index
  - but ultimately store them keyed by canonical `coco_category_id` from schema.

### Phase 4 — Recipe mining + portability

This is the user-facing “it must never silently break” area.

- [ ] Treat dataset schema as authoritative:
  - selected mining classes should be selected by `(schema id + canonical name)` pair.
- [ ] When attaching a CLIP head filter during mining:
  - compute CLIP head target index using schema + explicit mapping (not “best guess”).
  - if we can’t map, fail fast and show why (which name didn’t match, what the head contains).
- [ ] Extend recipe meta:
  - store `label_schema_hash` (or re-use `labelmap_hash` if we define it as schema hash)
  - store schema `classes` (already done as `labelmap`; consider renaming to `label_schema_classes`)
  - store `target_schema_id` and `target_schema_index` (for better debuggability)
- [ ] On apply:
  - if schema mismatch:
    - warn by default,
    - offer a “remap by name” option (best-effort), but never silently remap without user intent.

### Phase 5 — UI updates (reduce ambiguity)

- [ ] Everywhere the UI shows a class, show:
  - `name`
  - `schema id` (and optionally `schema index`)
  - do not label the number simply as “class id” without context
- [ ] In recipe mining UI:
  - surface schema mismatch warnings prominently
  - add a “debug” expandable section showing:
    - dataset schema hash
    - recipe schema hash
    - active CLIP head schema hash
    - for the selected class: dataset id/index, CLIP head index

## Full code-path inventory (things that must be audited/updated)

This section is intentionally exhaustive; treat it as a checklist for the refactor.

### Backend: `localinferenceapi.py`

**Dataset storage / metadata**
- [ ] `SAM3_DATASET_META_NAME = "sam3_dataset.json"` and helpers:
  - `_load_sam3_dataset_metadata`
  - `_persist_sam3_dataset_metadata`
  - any metadata “repair” code paths in `_list_all_datasets` / `_list_sam3_datasets`
- [ ] Qwen dataset metadata:
  - `_load_qwen_dataset_metadata`
  - `_persist_qwen_dataset_metadata`
  - `_ensure_qwen_dataset_signature`

**Dataset ingestion**
- [ ] `/datasets` zip upload path:
  - reads `labelmap.txt`, writes `sam3_dataset.json`
  - must also write `label_schema.json`
- [ ] `/qwen/dataset/*` (init/chunk/finalize):
  - ensure `dataset_meta.json` includes schema or references `label_schema.json`

**Dataset conversion**
- [ ] `_convert_yolo_dataset_to_coco`:
  - currently assumes `category_id = class_idx + 1`
  - must become schema-driven (even if schema keeps 1..N, the mapping must be explicit)
- [ ] `_convert_qwen_dataset_to_coco`:
  - currently uses `label_to_id = {label: idx+1}` from `meta["classes"]`
  - should map label strings via schema (name matching), not via “position in classes list”
- [ ] `_load_coco_index`:
  - currently merges and sorts categories by `id`
  - ensure it preserves canonical order and doesn’t silently reorder if schema says otherwise

**Recipe mining**
- [ ] `/agent_mining/jobs` creation path:
  - class selection should use schema ids/names consistently
  - ensure logging prints schema context
- [ ] CLIP head gating during mining:
  - `_find_clip_head_target_index` and call sites in mining loops
  - ensure mapping is schema-driven + explicit, not accidental
- [ ] Recipe save/load/export/import:
  - `_persist_agent_recipe`
  - `_load_agent_recipe`
  - `/agent_mining/recipes/import`
  - ensure schema fields are present and validated

**Prompt helper**
- [ ] `/sam3/datasets/{dataset_id}/classes`:
  - should return schema id + name + index (if meaningful)
- [ ] `_suggest_prompts_for_dataset`:
  - ensure category ordering returned matches schema canonical order
- [ ] preset save/load:
  - prompt helper presets keyed by class id; ensure class id is schema id

**SAM3 training**
- [ ] `/sam3/train/jobs` and config builder:
  - places using `cat_id = idx + 1` must be replaced with schema mapping
  - prompt variants mapping must support name/id/index

**Segmentation builder**
- [ ] `_plan_segmentation_build` + `_start_segmentation_build_job`:
  - ensure `classes` list and YOLO `class_idx` are consistent with schema
  - if source dataset has schema ids not equal to YOLO ids, add explicit mapping

**CLIP activation / inference**
- [ ] `/clip/train` (server-side wrapper around `tools/clip_training.py`):
  - ensure exported artifacts include schema metadata
- [ ] `/clip/active_model`:
  - validate classifier/labelmap vs schema
  - ensure UI sees and can display schema hash + ordering

### Tools (`tools/`)

These scripts aren’t all “production”, but they shape artifacts people reuse.

- [ ] `tools/clip_training.py`:
  - currently saves a label list but classifier order is sklearn-defined
  - must export schema + explicit mapping / reorder
- [ ] `tools/train_clip_regression_from_YOLO.py`:
  - CLI help claims it “preserves YOLO class order”; make that actually true
- [ ] `tools/clip_kmeans_and_regress.py`:
  - uses `sorted(set(labels))`; document or update to schema-driven ordering
- [ ] `tools/checkup_dataset.py`:
  - ordering checks should be schema-aware; prefer comparing by name rather than index
- [ ] `tools/reorder_labelmap.py`:
  - should output a schema mapping file, not only a reordered list
- [ ] `tools/detect_missclassifications.py`:
  - must be explicit about which label space it operates in; ideally schema-aware
- [ ] `tools/qwen_training.py`:
  - ensure run metadata includes schema hash and preserves class list order

### Frontend (`ybat-master/`)

- [ ] Anywhere classes are displayed/selected:
  - show schema hash + clarify id space
  - prefer name + schema id, not list indices
- [ ] Recipe mining UI:
  - show mapping debug info (dataset vs recipe vs CLIP head)
- [ ] Dataset upload UI:
  - if label schema exists, show it and allow exporting it

### Tests

- [ ] Add unit tests for `LabelSchema`:
  - hashing stability
  - name normalization + alias matching
  - schema inference from:
    - YOLO labelmap
    - COCO categories (non-contiguous ids)
    - legacy `classes` lists
- [ ] Add regression tests for mining mapping:
  - mapping a target class name/id to the right CLIP head probability index
  - ensure mismatches produce actionable errors (not silent 0 detections)

## Edge cases / design constraints

- **Non-contiguous COCO category ids**: must not assume 1..N.
- **Duplicate names** (case-insensitive or normalized collisions): must fail fast.
- **Aliases**: aliases can collide; canonical name must win.
- **Renames**: a rename is a schema change; we should treat it as incompatibility unless an explicit mapping is provided.
- **Merged datasets**: might contain overlapping but differently-ordered categories; avoid implicit merges.
- **Sklearn LogReg**:
  - `classes_` order is not under our control unless we enforce it via numeric labels or post-fit reordering.
  - if we post-fit reorder, must reorder `coef_`, `intercept_`, and any dependent attributes consistently.
- **Portability**: schema file should not embed absolute paths; only semantic class info.

## Complexity estimate (what this will take)

- **Moderate-to-high** complexity because it touches:
  - dataset ingestion,
  - dataset conversion,
  - CLIP training/export/activation,
  - SAM3 training,
  - recipe mining/apply,
  - UI displays.
- The actual implementation can be **incremental** if we:
  - introduce schema utilities first,
  - add validation + logging without changing behavior,
  - then migrate individual flows one by one.

## Suggested first “implementation sprint” checklist (minimal viable refactor)

If we want the biggest safety win with the least disruption:

1) Introduce `LabelSchema` + `label_schema.json` for datasets.
2) Make COCO conversion strictly schema-driven.
3) Make CLIP head export include explicit mapping and use it in mining.
4) Add UI debug panel for schema mismatches.

