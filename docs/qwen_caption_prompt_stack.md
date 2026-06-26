# Qwen Caption Prompt Stack Contract

This note records the captioning prompt-stack invariants that prevent the same
failure modes from returning during future prompt edits.

## Contract

- Prompt preview and live caption generation must use the same evidence builder.
- All reusable caption prompt layers must be visible in the Caption prompt stack
  and portable through caption recipes. The editable layers are: combined user
  request, main system prompt, detection context prompt, window/crop prompt,
  draft/refine prompt, window merge prompt, cleanup prompt, editor system
  prompt, coverage/count refinement prompt, and English rewrite prompt.
- Image-specific context is generated, not recipe-authored: image pixels,
  image tokens, boxes, counts, window bounds, matched full-frame object IDs,
  raw model outputs, and generated captions are visible in the complete prompt
  preview but must not be saved into portable recipes.
- Raw label names are backend identifiers. Model-facing prompt prose must use
  validated natural terms from the glossary or a safe naturalized fallback.
- Malformed glossary fragments such as `[`, `]`, `{`, `}`, empty text, or
  punctuation-only strings must never become class names, counts, box labels, or
  allowed-class text.
- Window captions, full-image composition, merge, cleanup, coverage, and
  language-rewrite prompts must receive the same class/count/glossary context.
- Window-local caption evidence must be reconciled back to full-frame annotation
  objects before full-image composition and merge prompts. Full-frame counts stay
  authoritative; window captions add close-up detail, not extra inventory.
- Internal object reference IDs in reconciliation text are prompt-only merge
  aids. They must never be presented as final caption content.
- Window location text must be generated from crop bounds, not crop center alone.
  Overlapping 2x2 crops should read as upper-left, upper-right, lower-left, and
  lower-right sections with percent bounds as prompt-only layout aids.
- Window caption sentence limits are explicit caption settings. They are not
  inferred from the final full-image caption length.
- Caption generation is high-cap rather than uncapped. Detailed/windowed calls
  may auto-lift output budgets, but loop detection and cleanup guardrails remain
  active.

## Caption Recipes

Caption recipes are browser-local until downloaded as JSON. A portable recipe is
allowed to contain reusable setup only:

- caption style text and opening phrase guidance
- every editable prompt-stack layer
- caption scope, windowing, model, and generation controls
- optional editable glossary text

Recipes must not contain image payloads, per-image boxes, label hints, used
counts, generated captions, prompt-preview placeholders, or backend image tokens.
Uploaded recipe JSON is applied to the current controls and saved locally only
after it passes this reusable-setup shape check.

The complete prompt-flow preview remains the source of truth for a specific
image. It combines recipe-controlled prompt layers with generated per-image
evidence and shows conditional templates for cleanup, coverage/count refinement,
and English rewrite guards.

## Regression Mechanisms To Avoid

- A malformed glossary term became the preferred broad term, producing prompt
  text like `broad term "["`, `COUNTS: [: 8`, and `{"label":"["}`.
- Center-based window wording made overlapping corner crops appear as
  `middle-center`, which misled the merge pass.
- Loose window observation text made repeated crop-edge detections look like
  separate objects. The reconciler must match window boxes to full-frame boxes
  whenever possible and mark unmatched evidence as context only.
- Merge and cleanup prompts previously received partial context and could lose
  glossary or count semantics after the first caption pass.
- Prompt preview assembled some sections separately from runtime, allowing the
  visible prompt and actual prompt to drift.
- Overlap deduplication guidance appeared twice in the preview merge template.

## Verification

Focused caption tests should assert that generated previews contain no malformed
bracket labels, preserve natural counts, expose cleanup and merge templates, use
bounds-aware window descriptions, and show reconciled full-frame/window object
evidence without turning window-only details into extra counts.
