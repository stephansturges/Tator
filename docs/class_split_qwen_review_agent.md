# Class Split Qwen Review Agent

The Class Split Explorer can ask a local Qwen VLM to review one likely-wrong
vignette before a human applies any label change. This is an advisory workflow:
Qwen never mutates labels directly.

## Anti-Drift Invariant

The VLM final judgment is the core of this workflow. Deterministic overlap,
scale, embedding, visual-quality, and cue checks are rails around the VLM
decision; they are not the default decision-maker. If Qwen finalization is
unstable, fix the model/runtime/schema/context/evidence path or expose an
explicit fallback mode. Do not silently replace the VLM review with a
controller-only skip/triage system and call it complete.

This subsystem follows the repo-level contract in [AGENTS.md](../AGENTS.md),
the durable project memory in [memory.md](../memory.md), and the governance notes
in [agent_governance.md](agent_governance.md). Any change to prompts, evidence
packing, guardrails, model routing, or benchmark interpretation must preserve
that contract and update this document in the same patch.

## User Flow

1. Run Class Split in all-class mode so likely-wrong candidates are available.
2. Optionally open `Qwen review context` above the vignettes and edit the class
   glossary or session guidance. Dataset-backed workspaces can save the glossary;
   transient workspaces send the text only with the current review request.
3. In the Likely wrong class toolbar, choose a Qwen reviewer model or leave it
   on the active model.
4. Click `Review with Qwen` on a vignette. If the vignette has a near-identical
   cross-class overlapping box, the button reads `Review dual bbox with Qwen` and
   the review switches to the narrower dual-box question.
5. The backend starts a bounded review job, renders evidence images, and polls
   the result back into the vignette.
6. The human still uses `Confirm current class`, `Switch class to ...`, `Skip`,
   or `See instance` to apply the decision.

## Backend Contract

Endpoint surface:

- `POST /class_analysis/jobs/{job_id}/points/{point_id}/qwen_review`
- `GET /class_analysis/qwen_review/{review_id}`
- `POST /class_analysis/qwen_review/{review_id}/cancel`
- `GET /class_analysis/qwen_review/{review_id}/evidence/{evidence_id}`

Each review writes artifacts under the parent Class Split job:

- `qwen_reviews/{review_id}/events.jsonl`
- `qwen_reviews/{review_id}/prompt_sources.json`
- `qwen_reviews/{review_id}/concept_briefs.json` when concept briefs are enabled
- `qwen_reviews/{review_id}/final.json`
- `qwen_reviews/{review_id}/evidence/*.jpg`
- `qwen_reviews/class_concept_briefs/*.json` and `*_examples.jpg` cached per
  parent Class Split job
- `qwen_reviews/class_pair_contrast_briefs/*.json` and `*_examples.jpg` cached
  per parent Class Split job when a current/suggested class pair is available

The final result schema is:

```json
{
  "decision": "confirm_current | accept_suggested | change_to_other | skip_uncertain",
  "target_class": "class name",
  "confidence": 0.0,
  "visual_quality": "clear | limited | poor",
  "object_visibility": "clear | partial | tiny_or_blurry | not_visible",
  "current_evidence": "strong | moderate | weak | none",
  "suggested_evidence": "strong | moderate | weak | none",
  "target_evidence": "strong | moderate | weak | none",
  "overlap_assessment": "none | duplicate_like | partial_contamination | target_contains_other | other_contains_target | near_context | unclear",
  "overlap_explains_candidate_similarity": false,
  "specificity_alignment": "supports_current | supports_suggested | supports_other | mixed | insufficient | not_applicable",
  "target_background_contrast": "target_specific | background_dominated | overlap_dominated | mixed | insufficient | not_applicable",
  "dual_bbox_resolution": "not_applicable | current_box_class | overlap_box_class | both_valid_overlapping_objects | uncertain_or_neither",
  "dual_bbox_conflict": {
    "enabled": true,
    "kind": "near_identical_cross_class_bbox",
    "review_mode": "dual_bbox_class_resolution",
    "current_class": "current class",
    "other_class_name": "overlapping box class",
    "other_point_id": "point id",
    "iou": 0.95,
    "target_area_covered": 0.98,
    "other_area_covered": 0.97
  },
  "anchor_evidence_current": "strong | moderate | weak | none",
  "anchor_evidence_suggested": "strong | moderate | weak | none",
  "local_context_evidence": "strong | moderate | weak | none",
  "local_consensus_evidence": "supports_current | supports_suggested | mixed | absent | not_applicable",
  "global_context_evidence": "strong | moderate | weak | none",
  "glossary_or_guidance_used": false,
  "visible_target_cues": ["visible target cue", "second cue when changing class"],
  "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_6"],
  "backend_visual_quality": {
    "tier": "clear | limited | poor",
    "bbox_width": 0.0,
    "bbox_height": 0.0,
    "crop_contrast": 0.0
  },
  "guardrail_reasons": [],
  "advisory_reasons": [],
  "guarded_recommendation": {
    "blocked": true,
    "decision": "accept_suggested",
    "target_class": "candidate class",
    "confidence": 0.82,
    "guardrail_reasons": ["why this cannot be applied automatically"],
    "rationale_short": "model rationale for the blocked recommendation"
  },
  "evidence_ids": ["target_context_1"],
  "rationale_short": "short reason",
  "counter_evidence": "what could make this wrong",
  "human_review_needed": true,
  "review_disposition": {
    "disposition": "actionable_class_change | dual_bbox_switch_overlap_class | dual_bbox_confirm_current | dual_bbox_both_valid_overlap | dual_bbox_unresolved | guarded_visual_quality | guarded_overlap_risk | guarded_missing_visible_cues | verified_no_class_change | no_actionable_opinion",
    "signal": "actionable | guarded_human_triage | useful_negative | no_signal",
    "label": "human-readable result label",
    "priority": "high | normal | low",
    "advisory_target_class": "class to inspect or apply",
    "primary_reason": "main controller reason"
  },
  "model_compact_arguments": {
    "decision": "skip_uncertain",
    "final_class": "class name recommended by the model",
    "confidence": 0.4
  },
  "expanded_by_controller": true,
  "controller_reconciliation": {
    "applied": false,
    "from_decision": "skip_uncertain",
    "to_decision": "accept_suggested",
    "reason": "why the controller reconciled a contradictory compact output"
  },
  "review_agent_controller": "state_machine_v2",
  "router": {
    "action": "finalize_now",
    "reason_code": "evidence_complete",
    "policy_reasons": []
  },
  "class_concept_briefs": {
    "enabled": true,
    "version": "class_visual_concept_brief_v3",
    "pair_contrast_version": "class_pair_contrast_brief_v2",
    "classes": ["current class", "suggested class"],
    "cache_keys": ["brief cache key"],
    "cache_hits": [false],
    "pair_cache_keys": ["pair contrast cache key"],
    "pair_cache_hits": [false]
  },
  "evidence_ledger": {
    "clean_visual_evidence_ids": ["target_context_1", "target_detail_2", "source_clean_3", "class_context_pack_6", "zoom_region_9"],
    "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"],
    "geometry_overlay_evidence_ids": ["source_overlay_4", "overlap_decomposition_5"],
    "local_consensus_evidence_ids": ["local_consensus_context_10"],
    "clean_visual_reference_evidence_ids": ["class_context_pack_6"],
    "dual_bbox_conflict": {
      "review_mode": "dual_bbox_class_resolution",
      "current_class": "current class",
      "other_class_name": "overlapping box class",
      "iou": 0.95
    },
    "policy": "visible_target_cues must come from clean visual evidence..."
  },
  "applied": false
}
```

`applied` must remain `false`; the frontend label-change controls are the only
place where the open dataset is changed. `guarded_recommendation` is advisory
only. It preserves the original model recommendation when backend guardrails
force a non-skip decision back to `skip_uncertain`, so benchmark audits and the
vignette UI can surface useful human-review signal without adding an automatic
mutation path.

## Agent Loop

The implementation now uses a controller-owned state machine rather than a
model-owned tool loop. This is intentionally closer to OpenClaw/Hermes practice:
the backend owns the state, decides which action schema is visible, records the
active schema in `events.jsonl`, and rejects outputs that do not match the
current state. The model never receives an open toolbox.

State sequence:

1. `required_evidence`: the controller renders fixed evidence before any model
   decision:
   `inspect_target_context`, `inspect_target_detail`, `inspect_source_overlay`,
   `inspect_overlap_decomposition`, `inspect_class_context_pack`,
   `inspect_same_image_scale_report`, `inspect_same_image_embedding_report`, and
   one clean `zoom_source_region` with `draw_bbox=false`.
2. `concept_briefs`: when `enable_class_concept_briefs` is true, the controller
   builds cached advisory visual concept briefs for the current class, suggested
   class, and at most one extra high-frequency neighbor class. It also builds one
   cached advisory pairwise contrast brief for the current-vs-suggested class
   pair when both classes are present. Class briefs are generated from clean
   trusted-but-diverse exemplar crops plus relevant glossary/guidance, saved
   under `qwen_reviews/class_concept_briefs/`, and injected only into the final
   prompt. The selector first filters for high-confidence non-suspicious anchors,
   then spreads examples across source images and projection space so Qwen sees
   real within-class variation instead of only the highest-scoring visual mode.
   Pair contrast briefs are generated from clean side-by-side trusted-diverse
   exemplars for both classes, saved under
   `qwen_reviews/class_pair_contrast_briefs/`, and focus on visible distinctions,
   shared ambiguity cues, and skip conditions.
   Cache keys include the model id, class name or pair names, glossary entries,
   review guidance, implementation version, and exemplar point ids. The UI
   enables this path for `Review with Qwen`.
3. `router`: local consensus is enabled by the UI/API default, but backend
   policy must still allow it before the model sees any consensus evidence. The
   only possible actions are `finalize_now` and
   `inspect_local_consensus_context`. If policy blocks local consensus, the
   controller skips this model call and routes directly to final.
4. `routed_optional_evidence`: if the router requested local consensus and the
   backend policy allowed it, the controller renders
   `inspect_local_consensus_context` itself and appends the observation.
5. `evidence_ledger`: the controller writes `evidence_ledger.json`, appends an
   `evidence_ledger` event, and injects one compact ledger message before final
   review. The ledger separates clean visual evidence from geometry overlays,
   local-consensus dot maps, and reference-only context. This follows the
   running-summary pattern used in agent/tool systems while keeping the summary
   backend-owned and auditable. Qwen must draw `visible_target_cues` from clean
   target/source pixels; overlay boxes, dot colors, labels, and neighbor counts
   may explain context but cannot be the sole basis for a class-change cue. For
   label-changing recommendations, Qwen must also emit
   `supporting_clean_evidence_ids` that point to the clean target/source evidence
   IDs supporting those cues.
6. `final`: the model receives a compact `finalize_review` schema and must
   return one plain JSON arguments object, not an outer tool envelope. Earlier
   MLX Qwen3.6 runs used an assistant prefix that pre-filled the outer
   `finalize_review` wrapper; benchmark logs showed this induced malformed
   `{}}` completions, so finalization now runs in no-prefix JSON mode. The
   final message keeps the clean target-detail, clean zoom, and class-context
   pack images; local-consensus views, deterministic reports, source overlays,
   and overlap-decomposition views remain compact text/ledger context so
   target-contained pixels stay central without hiding trusted examples. The backend
   expands that compact object into the full audit payload and applies
   guardrails. The compact schema includes an SDDF-inspired specificity audit:
   `specificity_alignment` states which class hypothesis is supported by
   target-contained object cues, and `target_background_contrast` states whether
   those cues are target-specific or dominated by background, overlap, or context.
   Any route call or evidence-tool call in this state is a schema
   failure. Retries remain allowed for malformed output; repeated failure
   becomes `skip_uncertain`. Degenerate repeated-token outputs are detected
   before JSON parsing and fail closed immediately as `skip_uncertain` so bad
   assistant text is never replayed into the next final attempt.

Required route schema:

```json
{
  "name": "route_review",
  "arguments": {
    "action": "finalize_now | inspect_local_consensus_context",
    "reason_code": "evidence_complete | needs_same_image_consensus | target_quality_not_clear | no_suggested_class | local_consensus_disabled | policy_blocked",
    "confidence": 0.0,
    "rationale_short": "short route reason"
  }
}
```

Model-facing compact final schema:

```json
{
  "decision": "confirm_current | accept_suggested | change_to_other | skip_uncertain",
  "final_class": "class name to apply",
  "confidence": 0.0,
  "visual_quality": "clear | limited | poor",
  "object_visibility": "clear | partial | tiny_or_blurry | not_visible",
  "current_evidence": "strong | moderate | weak | none",
  "suggested_evidence": "strong | moderate | weak | none",
  "target_evidence": "strong | moderate | weak | none",
  "overlap_assessment": "none | duplicate_like | partial_contamination | target_contains_other | other_contains_target | near_context | unclear",
  "overlap_explains_candidate_similarity": false,
  "specificity_alignment": "supports_current | supports_suggested | supports_other | mixed | insufficient | not_applicable",
  "target_background_contrast": "target_specific | background_dominated | overlap_dominated | mixed | insufficient | not_applicable",
  "dual_bbox_resolution": "not_applicable | current_box_class | overlap_box_class | both_valid_overlapping_objects | uncertain_or_neither",
  "local_consensus_evidence": "supports_current | supports_suggested | mixed | absent | not_applicable",
  "visible_target_cues": ["concrete cue from target/source pixels"],
  "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_6"],
  "counter_evidence": "short counter-evidence",
  "human_review_needed": true,
  "rationale_short": "visible facts only"
}
```

`final_class` is deliberately model-facing and means the label recommended by
the VLM, not "the current class of the target object." The controller still
stores the normalized backend result as `target_class` for compatibility. The
decision-to-class mapping is:

- `confirm_current`: `final_class` must be the current class.
- `accept_suggested`: `final_class` must be the suggested class.
- `change_to_other`: `final_class` must be a third labelmap class.
- `skip_uncertain`: `final_class` should normally be the current class.

The model is not asked to output anchor, scale, embedding, local-context, or
global-context bookkeeping fields. It sees those reports as evidence/context and
the controller writes the normalized audit fields after parsing the compact VLM
decision. In particular, `accept_suggested` normally
requires `anchor_evidence_suggested=strong`; a moderate suggested-anchor signal
is allowed only on the narrow clear-target path where target/source pixels,
local context, global context, overlap state, and weak current-class evidence
all agree. Otherwise the backend preserves the attempted recommendation as
`guarded_recommendation` and forces `skip_uncertain`. The model should not emit
glossary bookkeeping or arbitrary evidence ids. It should only cite clean
target/source ids in `supporting_clean_evidence_ids` for the specific visible
cues it claims.

Controller controls:

- required evidence writes `controller_tool_call` and `tool_result` events before
  the first model generation, so the clean zoom cannot be skipped by a brittle
  model-selected tool sequence
- controller evidence calls are not inserted into the transcript as assistant
  tool-call messages; Qwen sees observations and images only, preventing it from
  copying stale evidence tool names in the final state
- model calls are state-scoped: router sees `route_review`; final sees
  `finalize_review`; neither state sees arbitrary evidence tools
- final-state Qwen output is intentionally compact and generated without an
  assistant prefix; Python prefers `final_class`/`recommended_class` and still
  expands legacy aliases such as `target_class`, `target`, `class`, and
  `uncertain_class`, repairs common JSON-prefill fragments, records
  `model_compact_arguments`, and then validates the expanded result
- the active schema is logged as `tool_schema` and enforced by backend
  validation; on the current MLX-VLM path it is not passed through the
  chat-template `tools` argument because benchmark logs showed that path can
  produce bare `:` or prose instead of the JSON function call
- local consensus is enabled by the Class Split UI and Qwen review endpoint by
  default; backend policy still requires clear target quality, a suggested
  class, and no prior consensus evidence before the router can request it
- router requests that violate policy are coerced to `finalize_now` and logged
- overlap decomposition is always available and can return an empty/no-overlap
  section, so the model must actively reason about contamination instead of
  assuming the suggested class is correct
- concept and pairwise contrast briefs are optional advisory memory. They are
  generated from trusted examples and can improve class semantics, but the final
  prompt explicitly says fresh target pixels, clean source context, overlap
  evidence, and backend guardrails override them.
- final visual context is intentionally target-dominant: target-context and
  clean zoom/source images remain visible, while class context packs, consensus
  views, and overlays are retained as text/reference context only. This prevents
  clear anchor examples from pulling Qwen toward a class that is merely nearby
  or scene-compatible with the target.
- final-state schema or guardrail failures get one retry; a second failure
  returns `skip_uncertain`
- if Qwen emits a compact result that contradicts its own target class and
  visible-fact rationale, the controller may reconcile the action toward the
  text-supported class; this reconciliation is evidence-driven and has no
  hard-coded class-pair list
- every model call writes a `model_input` JSONL event with the exact message
  stack, active tool schema, model override, decode kwargs, and controller phase
  before the matching `model_output` event
- final results include `review_agent_controller=state_machine_v2` and the router
  result for auditability

The final prompt treats the current class and suggested class as hypotheses.
Qwen is explicitly allowed to use same-image anchors, wider-distribution
anchors, class glossary text, source context, and overlap decomposition together.
This is less blind than crop-only review, but still remains human-in-the-loop.

## SDDF Transfer Note

The SDDF paper, [Specificity-Driven Dynamic Focusing for Open-Vocabulary
Camouflaged Object Detection](https://arxiv.org/html/2603.26109v1), is relevant
because it addresses the same failure mode that appears in likely-wrong review:
target-object features can look very close to background or neighboring-object
features, so a model can over-weight context. SDDF uses fine-grained
sub-descriptions, removes noisy text components with SVD, contrasts object-region
and background-region similarities, and spatially focuses feature responses
toward discriminative target regions. The public
[SDDF repository](https://github.com/zh1fen/sddf) currently contains README/assets
and notes that training/inference code is coming later, so there is no code to
vendor directly.

Our first integration is therefore conceptual and audit-oriented rather than a
direct detector port. Qwen final review now emits:

- `specificity_alignment`: which class hypothesis is supported by
  target-contained object cues.
- `target_background_contrast`: whether the evidence is `target_specific`,
  `background_dominated`, `overlap_dominated`, mixed, or insufficient.

This keeps the VLM final judgment central while making the SDDF question
explicit: are we recommending a class change because the target itself has
discriminative features, or because background/overlap/context made the embedding
neighborhood look plausible? Class-change recommendations require the
target-specific path. Background-dominated or overlap-dominated recommendations
are preserved as guarded human-triage signals, not silently applied.

The deeper integration path is a learned or CLIP-aligned specificity evidence
module: generate per-class or per-instance sub-descriptions, decorrelate them,
score target-vs-background regions with a text-image aligned model, and feed that
score as another evidence tool before final Qwen review. DINOv3 crop embeddings
alone are not text-aligned enough for direct SDDF contrastive text-region scoring,
so that should be treated as a future model experiment rather than a small prompt
patch.

## Evidence Tools

`inspect_target_context` renders the target object crop with extra source-image
context and a bbox overlay. This is the primary visual evidence for the object.

`inspect_target_detail` renders the same target-centered context window without
any bbox overlay and deterministically enlarges small crops with Lanczos
interpolation. It is required before finalization so Qwen gets a clean close
view for visible target cues, but prompts and metadata state that no generated
detail has been added.

`inspect_source_overlay` returns two source-image views. The first is a clean
whole/source image with no boxes, used for scene layout and wider visual
context. The second is an annotated overlay that draws the target box plus
materially overlapping same-image boxes. Qwen should use the clean image for
class recognition and the overlay only for bbox geometry, overlap, and source
position.

`inspect_local_consensus_context` is opt-in optional second-stage evidence. It
returns one side-by-side local source view
around the target. The left panel is clean and has no annotation graphics. The
right panel uses center dots: orange for the target, blue for nearby same-image
current-class objects, and magenta for nearby same-image suggested-class
objects. This tool is designed to show local annotation consensus without
adding bbox graphics that become visual features. The dot map is not ground
truth; it can support or question a class hypothesis but cannot override unclear
target pixels. If this tool is not enabled or not inspected, the final result
uses `local_consensus_evidence=not_applicable`.

`inspect_overlap_decomposition` returns structured overlap metrics for
same-image boxes: IoU, target-area coverage, other-box coverage, relation label,
and a short interpretation. A common case is a target box crossing another
object's box: the target may contain texture from the other object, but that
contamination must not become the reason to relabel the target.

Near-identical cross-class boxes are handled as a separate `dual_bbox_conflict`
mode, not as generic contamination. The class-analysis result marks a conflict
when the target and another same-image box have different classes, IoU at least
0.90, and near-matching corners. Qwen then receives a narrower question:
resolve the target as the current class, the overlapping box class, both valid
overlapping objects with malformed geometry, or unresolved. This distinction is
important because some class pairs legitimately overlap in real datasets, such as
rider/vehicle, carried object/person, mounted object/support, or other
dataset-specific relationships from the glossary. The code must not hard-code
those pairs; Qwen must ground them in clean pixels, source context, glossary or
guidance, anchors, scale, embedding, and overlap evidence.

`inspect_class_context_pack` renders sectioned contact sheets:

- target object
- same-image current-class anchors when available
- same-image suggested-class anchors when available
- clean current-class anchors from the wider graph distribution
- clean suggested-class anchors from the wider graph distribution

Anchors are selected from high-confidence non-suspicious points first. They are
context, not ground truth; the validator still requires direct target evidence
before allowing a non-skip result. Context-pack crops do not draw bbox overlays;
class labels and scores are placed outside the crop pixels so the box graphics
do not become visual features. The rendered context pack remains saved for
human audit and model comparison, but final model vision no longer includes the
context-pack image; the final decision receives it as text/reference context
only.

`class_concept_briefs` are not evidence tools in the final state. They are a
controller-managed memory layer built before finalization when requested. For
each relevant class, the backend selects high-confidence non-suspicious examples
using the same class-neighborhood purity, crop geometry, outlier score, and
overlap-risk scoring used for trusted anchors. It then applies
`trusted_diverse_projection_v1` selection to include different source images and
different areas of the class projection. Qwen sees those clean exemplar crops
plus the relevant glossary entry and session guidance, then writes a compact JSON
class brief:

```json
{
  "class_name": "ClassA",
  "summary": "short advisory class concept",
  "visual_traits": ["visible cues from exemplars"],
  "valid_variations": ["accepted visual range"],
  "exclude_when": ["visible cues that argue against the class"],
  "common_confusions": ["nearby classes"],
  "uncertainty_triggers": ["when to skip"]
}
```

When a current/suggested class pair is available, the backend also renders
side-by-side trusted-diverse exemplars for the pair and asks Qwen for a compact
contrast brief:

```json
{
  "class_a": "CurrentClass",
  "class_b": "SuggestedClass",
  "summary": "short advisory pair distinction",
  "choose_class_a_when": ["visible cues for class A"],
  "choose_class_b_when": ["visible cues for class B"],
  "shared_or_ambiguous_cues": ["cues that are not decisive"],
  "hard_negative_cues": ["cues not allowed for switching"],
  "must_skip_when": ["pair distinction is not visible"]
}
```

The pair normalizer drops obvious-class bullets from `must_skip_when` when Qwen
misplaces them there, for example "object is clearly a car on a road". Skip
conditions are reserved for ambiguous, hidden, clipped, overlapping, or
contaminated evidence.

If generation fails or no exemplar sheet can be rendered, the backend writes a
deterministic fallback brief from the glossary/guidance and still completes the
review. Briefs are cached and reused inside the same parent Class Split job, but
they are invalidated by changing the model id, glossary entries, guidance text,
selected exemplars, class/pair names, or implementation version. This implements
visual in-context memory without letting the model teach itself unchecked
labels.

`zoom_source_region` renders additional source context around the target. One
clean call with `{"draw_bbox": false}` is mandatory before finalization, because
the agent must see wider object context without bbox graphics. Extra calls can
use `{"draw_bbox": true}` when the model specifically needs bbox geometry.

### Deterministic Same-Image Context

Two deterministic reports now run before finalization. They are generic: they
use only the active class label, source image id, bboxes, existing class-analysis
embeddings, and trusted-anchor selection scores. They do not contain any fixed
dataset class names or project-specific rules.

`inspect_same_image_scale_report` compares the target bbox geometry against
trusted same-current-class detections in the same source image. It reports
target bbox size, anchor count, anchor quantiles, ratios to anchor medians, and a
controller-owned signal:

- `supports_current`: target scale/aspect is close to same-image current-class
  anchors
- `questions_current`: target scale/aspect is a strong outlier and Qwen should
  reason about perspective before confirming the current class
- `neutral`: scale is neither a clear match nor a clear outlier
- `insufficient`: there are not enough trusted same-image anchors or bbox
  measurements

`inspect_same_image_embedding_report` compares the target embedding against
trusted same-current-class anchors in the same source image. It uses cosine
distances from the existing class-analysis embedding matrix and compares the
target-to-anchor distance distribution against the anchor-to-anchor distance
distribution. It reports the same signal vocabulary as the scale report.

These signals are controller-owned deterministic context, not model-authored
visual evidence. The compact final schema includes
`same_image_scale_evidence` and `same_image_embedding_evidence`, but the backend
normalizes them from the actual report metadata before validation so a model
cannot silently omit or contradict the deterministic result. The validator uses
the signals as advisory guardrails only:

- class-changing recommendations are confidence-capped when a deterministic
  report supports the current class
- confirmations are confidence-capped when a deterministic report questions the
  current class
- visible target cues still have to come from clean target/source pixels, never
  from a report signal alone

The embedding report keeps an in-process cache for the parent
`embeddings.npz`, keyed by parent job id, file mtime, size, and point count. It
does not write additional large sidecar files.

Backend guardrails are stricter than the prompt:

- target crops are measured for bbox pixel size, edge clipping, contrast,
  dynamic range, and a simple sharpness score
- poor target evidence is forced to persisted `skip_uncertain`
- any non-skip recommendation on a non-clear backend visual-quality tier is
  forced back to `skip_uncertain` and preserved as `guarded_recommendation`;
  limited-quality targets still get a final Qwen turn by default so humans can
  see Qwen's best advisory opinion without enabling automatic label
  recommendations when final model generation is enabled. Poor-quality targets
  skip Qwen by default because benchmark review showed mostly low-value/noisy
  advisory opinions; pass `allow_poor_final_review=true` or use the benchmark
  flag below only for explicit experiments.
- class-change recommendations (`accept_suggested` or `change_to_other`) require
  a clear backend visual-quality tier before they can become actionable
- final results include model self-check fields for quality, visibility,
  overlap, and class evidence strength; trusted-anchor, local-context,
  global-context, glossary, and evidence-id bookkeeping is added by the backend
  expansion layer
- final results may include `local_consensus_evidence`, which records whether
  same-image local consensus supports the current class, supports the suggested
  class, is mixed, absent, or not applicable; missing values are normalized to
  `not_applicable` when the optional tool was not used
- final results include controller-normalized `same_image_scale_evidence` and
  `same_image_embedding_evidence`. These fields mirror the deterministic
  report signals and are allowed to cap confidence, but cannot create visible
  target cues or override clean pixels.
- final results include an `evidence_ledger` summary so benchmark review can
  tell whether Qwen had clean visual, overlay/geometry, local-consensus, and
  reference evidence available at finalization time
- final results include `supporting_clean_evidence_ids`, a model-authored cue
  attribution list that must overlap clean target/source evidence for any
  `accept_suggested` or `change_to_other` recommendation; overlay-only,
  local-consensus-only, neighbor-count-only, or reference-only support is
  forced back to `skip_uncertain`
- visually adjacent or subtype-like class pairs are handled through the active
  dataset's glossary, review guidance, and generated pairwise contrast brief,
  never through fixed class names in code
- adjacent pairs are not automatically protected by forcing
  `current_evidence=strong`; the prompt tells Qwen to reserve strong current
  evidence for visible class-specific features, so clean subclass-like suggested
  examples can still become advisory class-change recommendations when the
  dataset-derived guidance supports it
- controller reconciliation is deliberately narrow and evidence-driven: it can
  only repair contradictory compact output when the target pixels, model text,
  evidence fields, visual quality, and overlap state agree
- `accept_suggested` requires strong suggested-class target evidence and strong
  suggested-class anchor agreement based on target-contained features. Moderate
  suggested-anchor agreement is allowed only for a clear target, weak/none
  current evidence, strong local and global context, and no material overlap; it
  stays confidence-capped and human-review-needed. `accept_suggested` is forced
  to `skip_uncertain` if Qwen also reports `current_evidence=strong`
- `accept_suggested` and `change_to_other` require at least two concrete
  `visible_target_cues`. The validator removes class-label-only boilerplate such
  as "matches suggested class", so a class-change recommendation has to expose
  positive target-pixel descriptors rather than class names, neighbor labels, or
  pure scene context.
- Negative or absence claims, color-only claims, and viewpoint, location, or
  background-only phrases such as overhead perspective, nearby context,
  shadows, or generic scene placement can remain in the reasoning, but they do
  not count toward the visible-cue threshold for a class-changing recommendation.
  A change still needs positive target-contained evidence.
- Those class-changing decisions also require `supporting_clean_evidence_ids`
  tied to clean target/source evidence. Clean reference packs can inform the
  comparison, but they cannot be the only evidence cited for visible target
  cues.
- `accept_suggested` is also forced to `skip_uncertain` when local consensus
  supports the current class and Qwen does not report current evidence as weak
  or none
- `confirm_current` requires at least moderate current-class anchor evidence
- `confirm_current` is forced to `skip_uncertain` if Qwen also reports
  `suggested_evidence=strong`
- `confirm_current` is also forced to `skip_uncertain` when local consensus
  supports the suggested class and Qwen reports strong suggested evidence
- if overlap contamination explains why the target looks similar to the
  suggested class, the validator forces `skip_uncertain`
- class-change recommendations are forced to `skip_uncertain` when overlap is
  `duplicate_like`, `target_contains_other`, `other_contains_target`, or
  `unclear`; `partial_contamination` is allowed only for clear targets when
  suggested-class anchor evidence, local context, and global context are all
  strong and Qwen's rationale explicitly explains why the overlap does not
  account for the target's visible class features
- the narrow exception is `dual_bbox_conflict`: if a near-identical cross-class
  box is detected, `duplicate_like` overlap can pass only when Qwen sets
  `dual_bbox_resolution=overlap_box_class`, the recommended `target_class` is
  exactly the overlapping box's class, the target/source evidence is clear and
  strongly target-contained, and all normal clean-evidence requirements pass.
  Third-class jumps remain blocked in dual-bbox mode.
- `dual_bbox_resolution=both_valid_overlapping_objects` is represented as
  `skip_uncertain` plus a `dual_bbox_both_valid_overlap` review disposition,
  because that means the human should inspect/fix geometry rather than simply
  relabel the target.
- the validator has no hard-coded class aliases or dataset-specific subtype
  rules. Subtype ambiguity must come from the active labelmap, glossary, review
  guidance, generated concept briefs, or generated pairwise contrast briefs.
- cue normalization is deliberately domain-generic. It may remove protocol
  boilerplate, negations, class labels, color-only cues, and background-only
  phrases, but it must not carry a committed list of object parts for the current
  benchmark dataset. Class/object vocabulary belongs in the labelmap, glossary,
  user guidance, or model-generated concept/pair briefs.
- `skip_uncertain` confidence is capped at 0.50, and hard-guardrail skips are
  capped lower, so skipped reviews do not look like high-confidence class
  recommendations
- the boxed `target_context` crop uses nearest-neighbor zooming for human-visible
  pixel fidelity, while the clean `target_detail` crop uses deterministic
  Lanczos enlargement for model readability. The prompt explicitly says this is
  interpolation, not generated super-resolution.
- images sent to the VLM are bounded to the backend
  `CLASS_ANALYSIS_QWEN_REVIEW_MODEL_IMAGE_MAX_SIDE` limit, while saved evidence
  artifacts remain full-size. This protects MLX/Metal runs from oversized visual
  prompts without hiding the audit evidence from humans.
- clean source/context evidence is provided separately from annotated overlays
  so the model can reason about visual class features without bbox graphics
  interfering
- clean zoomed source context is a required evidence gate, not an optional
  model preference
- production code must not contain project-specific class heuristics. Default
  glossaries are naturalized label names only; richer semantics have to come
  from the active dataset glossary, user guidance, retrieved examples, or
  generated concept/pair briefs.

### Current Mac/MLX Finalization Policy

`CLASS_ANALYSIS_QWEN_REVIEW_ENABLE_MLX_FINAL` defaults to `true`. The VLM final
decision is the core of the review flow on Mac: the controller renders the fixed
evidence pack, sends clean target/detail/zoom/class-context images to the
selected Qwen reviewer, and parses one compact `finalize_review` JSON object.
The deterministic scale, embedding, overlap, and anchor reports are rails and
audit context, not replacements for the VLM.

The current preferred local test model is
`vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit`. The Heretic MLX variant
was not exposed as a default reviewer after local smoke tests produced invalid
text in this harness. If future MLX command-buffer failures recur, use
`CLASS_ANALYSIS_QWEN_REVIEW_ENABLE_MLX_FINAL=false` only as an explicit fallback
debug mode; do not treat controller preflights as the main product behavior.

The model-facing final schema uses `final_class`, not `target_class`, because
the latter caused Qwen to confuse "class of the reviewed target object" with
"label to apply." The backend still accepts old `target_class` outputs and
stores normalized results as `target_class` for UI/API compatibility.

Guardrails can still block automatic mutation. In that case the final result is
`skip_uncertain`, but `guarded_recommendation` and `review_disposition` preserve
the VLM's recommendation as a first-class human-triage signal. The UI should
display these as guarded suggestions and preselect the suggested target class
for manual reassignment.

For moderate-anchor class changes without same-image scale/embedding support,
the backend runs a second VLM cue-verifier pass. That pass must inspect the
clean target/source evidence and explicitly set `current_class_plausible`.
If Qwen says the current class is still plausible, the final result remains
human-triage `skip_uncertain`; the raw guarded recommendation and verifier
reason are preserved in the payload.

The same verifier also adjudicates the moderate anchor itself. It must return
`anchor_support_verified`, `anchor_support_basis`, and `anchor_support_reason`.
Moderate suggested-anchor recommendations can become actionable only when Qwen
sets `anchor_support_basis=target_specific_anchors`, verifies no direct
current-class target pixels remain, and either no material overlap is present or
the overlap is separately rebutted from clean target/source pixels. Shared broad
shape, color, size, position, background, or context matches use
`shared_generic_anchors` and remain guarded human-triage output. This keeps the
promotion path VLM-centered: the controller validates quality, overlap, clean
evidence ids, and mutation safety, but Qwen owns the visual anchor sufficiency
judgment.

The cue-verifier pass may not promote a class change from shared generic cues
alone. If Qwen reports `current_class_plausibility_basis=shared_generic_cues`,
the promotion needs an independent same-image scale, same-image embedding, or
local-consensus signal questioning the current class. This prevents top-down
shape, color, position, or flat-surface language from turning into a class
change when the same target pixels could still plausibly fit the current label.
If the verifier reports shared generic current-class plausibility while also
reporting no current-class positive cues and verified target-specific anchors,
the backend treats that as an inconsistent wording issue and lets the normal
validator decide from the explicit cue and anchor fields.

Targeted verifier probe after adding anchor-support adjudication:

- `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/anchor_verifier_probe3_recordfix_3_1780884420.json`
  reran three clear-tier moderate-anchor guarded rows with the 35B abliterated
  MLX reviewer, local consensus, class concept briefs, and per-review
  subprocess isolation.
- Result: 3/3 completed, 0 failures, 0 final validation errors, 0 unsafe audit
  issues, and schema sequence `finalize_review->verify_visible_cues->verify_visible_cues`
  for all rows.
- The verifier returned `anchor_support_basis=target_specific_anchors` for all
  three rows. Two previously guarded rows promoted to actionable
  `accept_suggested`; one stayed guarded because overlap still remained too
  entangled after validation.
- This is not a replacement for the 100-row benchmark. It proves the new
  moderate-anchor verifier path is live, auditable, and capable of increasing
  useful action rate on the exact failure mode it targets.
- Focused clear-tier guarded validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/anchor_verifier_clear_guarded14_14_1780886770.json`
  used `--source-backend-tier clear --source-guarded-only --count 14` against
  the prior 100-row source run, with the same 35B abliterated MLX reviewer,
  local consensus, class concept briefs, and per-review subprocess isolation.
  Result: 14/14 completed, 0 unsafe audit issues, 4 actionable
  `accept_suggested` recommendations, 10 guarded recommendations, and 14
  effective human signals. The cue verifier ran on 9 rows, verified
  target-specific anchors on those 9 rows, and promoted 4 previously guarded
  model opinions. The visual sheets are
  `anchor_verifier_clear_guarded14_14_1780886770_visual_non_skip.jpg`,
  `anchor_verifier_clear_guarded14_14_1780886770_visual_guarded.jpg`, and
  `anchor_verifier_clear_guarded14_14_1780886770_visual_all.jpg`.
- Interpretation: the anchor verifier increases action rate on clear
  target-specific anchor cases without replacing Qwen's final judgment. The 10
  remaining guarded rows are still useful human-triage signals, mostly blocked
  by overlap risk, missing visible cues, or current-class overlap evidence.
- Prompt-contract follow-up: the same run showed `cue_verifier_parse_error` on
  all 9 verifier calls before the repair turn. The repair responses were valid,
  so the issue was not missing visual reasoning; the first-pass verifier prompt
  was too loose and let Qwen return legacy keys such as `current_class`,
  `proposed_target_class`, or `verified_evidence_ids` while omitting required
  fields. The first verifier prompt now mirrors the repair contract: exact
  required keys, a JSON skeleton, no extra legacy keys, and direct instruction
  to use `supporting_clean_evidence_ids`. Benchmark summaries and audits now
  report `cue_verifier_parse_error_count` directly so future runs can measure
  whether verifier protocol quality improves.
- Prompt-contract rerun:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/cueverifier_promptfix_clear_guarded14_14_1780888993.json`
  reran the same 14 clear-tier guarded source rows after tightening the first
  verifier prompt. Result: 14/14 completed, 0 failures, 0 unsafe audit issues,
  0 `cue_verifier_parse_error` events, 4 actionable `accept_suggested`
  recommendations, 10 guarded recommendations, and 14 effective human signals.
  The cue verifier ran on 9 rows, promoted 4 rows, and now used the direct
  `finalize_review->verify_visible_cues` sequence instead of requiring a second
  verifier repair turn.
- Visual audit note and follow-up hardening: the actionable sheet
  `cueverifier_promptfix_clear_guarded14_14_1780888993_visual_non_skip.jpg`
  showed one visually arguable `Truck -> LightVehicle` case at crop scale. The
  issue was not a dataset-specific label rule; it was a generic visual-reasoning
  failure mode where Qwen accepted a class change while its own text still said
  the current class was plausible, and while the proposed class explanation
  focused on a recognizable subpart rather than forcing a whole-bbox/object
  extent check. The final `finalize_review` schema and the cue verifier now
  require `whole_target_extent_supported` and `whole_target_extent_reason`, and
  the validator blocks `accept_suggested`/`change_to_other` when the model's own
  visible-fact text supports the current class. Focused rerun:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/textconflict_narrowed_suspect1_1_1780895831.json`
  converted the exact row from actionable `accept_suggested` to
  `skip_uncertain` with a preserved guarded recommendation, 0 unsafe audit
  issues, 1 effective human-triage signal, and
  `whole_target_extent_counts={"supported": 1}`. This preserves the VLM's
  recommendation for the reviewer while preventing a self-contradictory VLM
  class-change signal from becoming actionable.

Latest Mac probe after restoring VLM finalization and the `final_class` schema:

- `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vlm_finalclass_vanch007_probe2_2_1780820178.json`
  ran two limited-quality likely-wrong vignettes through the 35B abliterated
  MLX reviewer with `max_new_tokens=1000`.
- Result: 2/2 completed, 0 final validation errors, one `finalize_review` schema
  call per row, 2 guarded VLM recommendations, and 2 effective human-triage
  signals.
- Both raw model outputs were coherent `accept_suggested` recommendations with
  `final_class=LightVehicle` and concrete visible cues. The controller blocked
  automatic mutation because the crops were limited and/or overlap/scale rails
  made them unsafe for automatic relabeling.

## Glossary And Guidance

The frontend sends two text fields with each review:

- `labelmap_glossary`: class semantics. When a linked backend dataset is open,
  this is loaded from `/datasets/{dataset_id}/glossary`; otherwise it falls back
  to one `ClassName: ClassName` line per current class.
- `review_guidance`: session-only policy. Use it for dataset-specific intent,
  such as broad obstacle definitions, intentional class merges, or known
  ambiguous cases.

The backend normalizes both fields, writes them to `review_context.json`, and
injects them into the initial user prompt. They are advisory: a final decision
still has to satisfy the validator.

## Benchmark Audit Gate

The canonical V1 baseline is documented in
[class_split_qwen_review_v1_benchmark.md](class_split_qwen_review_v1_benchmark.md).
Use it as the comparison point for future Qwen reviewer changes: a candidate
must report automated audit counts and manual visual precision on actionable
rows, not only raw `skip_uncertain` totals.

Use the benchmark runner with `--audit` after any agent, prompt, evidence,
model, or validator change. Run this from an activated project environment
because the benchmark runner imports the backend:

```bash
source .venv-macos/bin/activate
python tools/run_class_split_qwen_review_benchmark.py \
  --job-id ca_c5c4a7d6ea \
  --source-run uploads/class_analysis/<job-id>/qwen_reviews/<prior-run>.json \
  --count 100 \
  --run-label qwen_agent_candidate \
  --model-id vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit \
  --enable-local-consensus \
  --enable-class-concept-briefs \
  --audit \
  --fail-on-unsafe
```

The benchmark runner reviews limited-quality targets by default for human-triage
signal, matching the app path. Add `--skip-limited-final-review` when
intentionally reproducing the older controller behavior that skipped final Qwen
review for limited targets. Add `--allow-poor-final-review` only for explicit
poor-evidence experiments; poor targets otherwise skip Qwen because the visual
audit found little useful signal there.

When a change targets a narrow failure mode, filter a prior source run before
sampling instead of taking the first `N` rows. The runner supports
`--source-backend-tier`, `--source-decision`, `--source-disposition`,
`--source-disposition-signal`, `--source-guarded-only`, and
`--source-reviewable-only`; repeat an option or pass comma-separated values.
Filtering happens before `--start` and `--count`, so a command such as
`--source-backend-tier clear --source-guarded-only --count 14` evaluates clear,
reviewable disagreement rows rather than measuring mostly poor or limited
targets. This matters for usefulness work: broad 100-row gates catch safety and
runtime failures, while focused filtered runs show whether a specific VLM rail
actually turns guarded model opinions into better advisory actions.

Each run writes three visual audit sheets when matching rows exist:
`*_visual_non_skip.jpg` for actionable recommendations, `*_visual_guarded.jpg`
for blocked model opinions, and `*_visual_all.jpg` for the sampled review set.
Use `review_disposition_signal_counts` and `effective_human_signal_count` to
measure whether the agent produced useful triage work. Raw `skip_uncertain`
counts are intentionally conservative and include guarded model suggestions
that are displayed for human review but blocked from automatic relabel advice.

For an existing run, the lightweight analyzer can use system Python:

```bash
python3 tools/analyze_class_split_qwen_review_benchmark.py \
  uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/<run>.json \
  --write-json uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/<run>_audit.json \
  --fail-on-unsafe
```

The audit script checks the safety invariants that should not regress:

- no actionable non-skip recommendation on limited or poor target evidence
- no class-changing recommendation on non-clear backend visual quality
- no `accept_suggested` decision when Qwen reports strong evidence for the
  current class
- no class-changing recommendation when either the final schema or the
  second-pass cue verifier says the clean target/source pixels still plausibly
  fit the current class
- no `accept_suggested` decision without strong suggested-class anchor
  agreement, except the narrow clear-target path where moderate anchor agreement
  is confidence-capped, all other target/source evidence is strong, and Qwen has
  explicitly checked `current_class_plausible=false`
- no `confirm_current` decision when Qwen reports strong suggested-class
  evidence
- no class-changing recommendation without at least two concrete target-visible
  cues and clean target/source evidence ids supporting those cues
- no class-changing recommendation for duplicate-like, contained, or unclear
  overlap states
- no class-changing recommendation on `partial_contamination` unless the model
  explicitly says why overlap does not explain the target's visible features
- no class-changing recommendation when the overlap decomposition shows that
  the current class dominates the target bbox and the suggested class appears
  only as weak overlap/context

The same report also lists guarded clear-target candidates: rows where the
target was inspectable and the model tried to produce an actionable result, but
the backend forced or kept a skip. Those rows are useful for improving
usefulness, but they are not failures by themselves; the current workflow
deliberately prefers missed automation over bad label-change advice.

For dual-bbox changes, the audit report must expose
`dual_bbox_resolution_counts` in addition to disposition counts. This matters
because the overlap-specific flow is valuable only when near-identical
cross-class boxes are resolved by the VLM as `overlap_box_class` or explicitly
left for human geometry review as `both_valid_overlapping_objects` /
`uncertain_or_neither`.

## Validation Evidence

Historical real-model validation was run against the completed all-class Class
Split job `ca_c5c4a7d6ea`, first using
`nightmedia/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx` and later
the experimental `vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit`.
Class names mentioned in the validation notes are local benchmark labels only.
They are not implementation rules; the runtime controller relies on the active
labelmap, glossary, review guidance, generated concept briefs, generated pairwise
contrast briefs, evidence fields, and overlap state.

- Genericity V4 100-crop gate:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/genericity_v4_wide100_100_1780875777.json`
  reran the same 100-row benchmark after removing benchmark-specific cue
  vocabulary from the validator and adding the shared-generic-cue promotion
  guard. It completed 100/100 in 858.0 seconds with 0 backend failures, 0 final
  validation errors, 0 unsafe audit issues, 4 actionable `accept_suggested`
  recommendations, 66 guarded recommendations, and 70 effective human signals.
  The cue verifier ran on every row, used the repair path on 11 rows, and
  promoted one guarded recommendation. Compared with
  `genericity_v3_wide100_100_1780874346.json`, the only decision change was
  `#83 Truck->Building`, which was demoted from `accept_suggested` to guarded
  `skip_uncertain` because Qwen's supporting cues were shared generic geometry
  without same-image scale, embedding, or local-consensus support. Manual spot
  checks of the four remaining actionables found them plausible human-reviewed
  class-change suggestions: `#26 Truck->Building`, `#47 LightVehicle->Building`,
  `#89 Truck->Building`, and `#92 Building->Solarpanels`.
- SDDF/dual-bbox V3 100-crop gate:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/sddf_dual_v3_wide100_100_1780857344.json`
  reran the V1 100-row benchmark with SDDF-style target/background fields,
  near-identical dual-bbox conflict mode, compact-output overlap normalization,
  and the repaired benchmark/audit recorder. Result: 100/100 completed, 0
  backend failures, 0 final validation errors, 4 actionable
  `accept_suggested` recommendations, 66 guarded recommendations, 70 effective
  human signals, and 0 unsafe audit issues in
  `sddf_dual_v3_wide100_100_1780857344_audit.json`.
  `dual_bbox_resolution_counts` were 93 `not_applicable` and 7
  `overlap_box_class`; two of the four actionable recommendations were
  `dual_bbox_switch_overlap_class` cases. Compared with the V1 benchmark,
  Qwen now promotes two additional clear dual-bbox/building corrections, while
  the previously questionable Truck to LightVehicle action is demoted to a
  guarded skip because the clean visible cue verifier found only one concrete
  target-class cue. Manual review of
  `sddf_dual_v3_wide100_100_1780857344_visual_non_skip.jpg` found the four
  actionable recommendations visually plausible and all still advisory/human
  controlled.
- Overlap-verifier V5 100-crop gate:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/overlap_verifier_v5_wide100.json`
  reran the same 100 rows after adding the first visible-cue verifier. It
  completed 100/100 with 0 backend failures, 0 final validation errors, 0 unsafe
  audit issues, 7 actionable `accept_suggested` recommendations, 63 guarded
  recommendations, and 70 effective human signals. Manual visual inspection of
  `overlap_verifier_v5_wide100_visual_non_skip.jpg` found that two actions were
  too aggressive (`#41 Truck->Building`, `#83 Truck->Building`) and two
  `Gastank->Building` actions (`#33`, `#40`) depended on project-specific class
  semantics. The failure mode was not a missing controller signal; it was that
  Qwen could accept the suggested class while the current class still remained
  visually plausible.
- Current-class-plausibility V2 100-crop gate:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/plausibility_v2_wide100.json`
  reran the V5 rows with a required second-pass cue verifier for moderate-anchor
  class changes that lack same-image scale/embedding support. The verifier must
  explicitly answer whether the clean target/source pixels still plausibly fit
  the current class. The run completed 100/100 in 872.8 seconds with 0 backend
  failures, 0 final validation errors, 0 unsafe audit issues, 3 actionable
  `accept_suggested` recommendations, 67 guarded recommendations, 70 effective
  human signals, 11 cue-verifier calls, 0 verifier promotions, and 10
  verifier-reported current-class-plausible cases in
  `plausibility_v2_wide100_audit.json`. Compared with
  `overlap_verifier_v5_wide100.json`, exactly four decisions changed:
  `#33 Gastank->Building`, `#40 Gastank->Building`, `#41 Truck->Building`, and
  `#83 Truck->Building` were demoted from `accept_suggested` to guarded
  `skip_uncertain` with VLM-generated current-class plausibility reasons. The
  remaining actionables stayed `#47 LightVehicle->Building`,
  `#89 Truck->Building`, and `#92 Building->Solarpanels`; visual inspection of
  `plausibility_v2_wide100_visual_non_skip.jpg` found those three defensible.
  Individual spot checks of the four demoted rows confirmed the new behavior:
  each target is ambiguous enough that automatic relabeling would be too
  aggressive, while the guarded VLM recommendation remains useful human triage.
- Pre-hardening 100-crop run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_validation_100_1780605874.json`
- Result: 100/100 review jobs completed, 0 backend failures, 55
  `accept_suggested`, 32 `confirm_current`, 13 `skip_uncertain`,
  `applied=false` for all records.
- Issue found: the model called `inspect_overlap_evidence` in 99/100 records
  even when no overlap evidence was required, and 48/100 records contained prose
  or repeated-text model turns after evidence collection.
- Post-hardening 10-crop protocol smoke:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_finalonly_validation_10_1780609761.json`
- Result: 10/10 completed, all in 4 model turns, 0 extra overlap calls, around
  15 seconds per review.
- Post-conservative-prompt 5-crop smoke:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_conservative_validation_5_1780610076.json`
- Result: 5/5 completed, all in 4 model turns, 0 extra overlap calls, around
  15 seconds per review.
- Quality-gate targeted 10-crop run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_qualitygate_targeted10_1780639114.json`
- Result: 10/10 completed, 0 extra overlap calls, 9 `skip_uncertain`, 1
  `accept_suggested`. The targeted set included previous manual problem cases;
  the poor/limited cases were downgraded to safe skips.
- Quality-gate 30-crop run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_qualitygate_validation_30_1780639686.json`
- Result: 30/30 completed, 0 backend failures, 0 extra overlap calls, 24
  `skip_uncertain`, 5 `accept_suggested`, 1 `confirm_current`.
- Manual image audit of that 30-crop run found one accepted
  `Truck`/`LightVehicle` construction-vehicle case that should remain human
  review, so the validator was tightened for class-adjacent pairs and
  context-reliant rationales.
- Replay of the same 30 model outputs through the tightened validator:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_qualitygate_validation_30_1780639686_replayed_latest_validator.json`
- Replayed result: 28 `skip_uncertain`, 1 `accept_suggested`, 1
  `confirm_current`. A 4-crop live probe
  `real30b_qualitygate_probe4_1780641025.json` confirmed the problematic
  `Truck`/`LightVehicle` case is now forced to `skip_uncertain`, while a clear
  `Truck` confirmation still passes.
- Clean-zoom-required controller run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_cleanzoom_controller_validation_30_1780652060.json`
- Result: 30/30 completed, 0 backend failures, every review had
  `model_input` and `model_output` JSONL events, every review had the
  controller-rendered clean `zoom_source_region(draw_bbox=false)` evidence, and
  every review completed as `skip_uncertain`.
- Interpretation: the controller/tool harness is now stable, but the advisory
  decision path is very conservative. Visual audit sheets
  `real30b_cleanzoom_controller_validation_30_1780652060_visual_all.jpg` and
  `real30b_cleanzoom_controller_validation_30_1780652060_visual_clear.jpg`
  show many defensible skips for tiny, contaminated, or class-adjacent targets,
  but also clear targets where Qwen proposed a plausible accept/confirm and the
  validator forced `skip_uncertain`.
- Advisory-validator 30-crop run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_advisory_validation_30_1780654664.json`
- Result: 30/30 completed, 0 backend failures, every review had clean zoom,
  controller-required calls, and model input/output logs. Decisions were 7
  `accept_suggested`, 4 `confirm_current`, and 19 `skip_uncertain`.
- Visual audit found this was no longer a skip-all harness, but class-change
  recommendations were still too permissive when Qwen also reported strong
  evidence for the current label or when the backend quality tier was limited.
- Hardened 30-crop live run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_hardened_validation_30_1780656637.json`
- Result: 30/30 completed, 0 backend failures, every review had clean zoom,
  controller-required calls, and model input/output logs. Decisions were 7
  `accept_suggested`, 2 `confirm_current`, and 21 `skip_uncertain`; protocol
  errors fell to 3 records and there were no high-confidence guarded skips.
- Strict replay of that hardened run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/real30b_hardened_validation_30_1780656637_replayed_strict_validator.json`
- Replayed result: 3 `accept_suggested`, 1 `confirm_current`, 26
  `skip_uncertain`, 0 high-confidence skips, 0 high-confidence guarded skips.
  Visual audit sheet
  `real30b_hardened_validation_30_1780656637_strict_visual_non_skip.jpg`
  leaves only defensible advisory recommendations: two clear LightVehicle
  corrections, one Building correction, and one low-risk confirm-current case.
- Compact-state smoke run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/state_machine_compact_smoke_5_1780675832.json`
- Result: 5/5 completed, 0 backend failures, 0 final validation errors. The
  compact final schema fixed the full-schema collapse, but all 5 outputs were
  `skip_uncertain`, including one clear local benchmark case where Qwen's own
  compact evidence favored the suggested class.
- Compact-state reconciliation smoke:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/state_machine_compact_reconcile_smoke_5_1780676214.json`
- Result: 5/5 completed, 0 backend failures, 0 final validation errors, 1
  `accept_suggested`, 4 `skip_uncertain`. The accepted case was a clear
  contradictory skip reconciled by the controller from evidence fields and
  visible-fact text.
- Compact-state 30-crop validation run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/state_machine_compact_reconcile_validation_30_1780676412.json`
- Result as originally run: 30/30 completed, 0 backend failures, 5
  `accept_suggested`, 25 `skip_uncertain`, 9 validation errors recovered by the
  retry path, and 3 controller-reconciled accepts.
- Visual audit found two reconciled accepts were bad promotions from weak
  context-only cues. The reconciliation rail was then narrowed to evidence-field,
  text-consistency, visual-quality, and overlap checks rather than class names.
  Replaying the raw compact outputs through the current parser/controller yields
  4 `accept_suggested`, 26 `skip_uncertain`, 1 evidence-driven reconciliation,
  and one unrecovered malformed output.
- Final compact smoke after parser repair and narrowed reconciliation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/state_machine_compact_final_smoke_5_1780677626.json`
- Result: 5/5 completed, 0 backend failures, 0 validation errors, 1
  `accept_suggested`, 4 `skip_uncertain`, and 1 evidence-driven controller
  reconciliation.
- Heretic 35B-A3B MLX compatibility smoke:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/heretic35b_mlx_smoke_1_1780678924.json`
  failed on the pinned MLX-VLM runtime because `qwen3_5_moe` is unsupported.
  A temporary MLX-VLM 0.6.1/Transformers 5 overlay with local shims reached
  generation, but `heretic35b_fullshim_smoke_1_1780681431.json` produced
  invalid text and the controller downgraded the single review to
  `skip_uncertain`. The checkpoint is therefore tracked as a candidate but not
  exposed as a selectable reviewer.
- Working Qwen3.6 35B-A3B MLX smoke:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_integrated_smoke_5_1780684758.json`
  used `vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit` through the
  normal `.venv-macos` path with MLX-VLM 0.6.1 and the local split-expert
  compatibility shim. Result: 5/5 completed, 0 backend failures, 0 final
  validation errors, 1 `accept_suggested`, 4 `skip_uncertain`. Visual audit of
  `vanch007_qwen36_integrated_smoke_5_1780684758_visual_all.jpg` found the
  accepted Truck to LightVehicle case defensible and the skipped cases tiny,
  clipped, or ambiguous.
- Qwen3.6 35B-A3B 30-crop validation before the strict overlap rail:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_vehicle_guard_validation_30_30_1780686690.json`
  completed 30/30 reviews with 0 backend failures and 0 final validation
  errors. Manual visual audit found one ambiguous same-domain local benchmark
  case was correctly forced to `skip_uncertain`, but another overlap-heavy
  accept was not safe because the target crop was materially entangled with
  overlapping boxes and mostly showed context rather than a clean object.
- Qwen3.6 35B-A3B strict-overlap validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_strict_overlap_validation_30_30_1780687541.json`
  completed 30/30 reviews with 0 backend failures, 0 final validation errors,
  2 `confirm_current`, 28 `skip_uncertain`, and 0 class-changing accepts.
  Visual audit of
  `vanch007_qwen36_strict_overlap_validation_30_30_1780687541_visual_non_skip.jpg`
  found the two confirmations defensible: one Solarpanels confirmation where
  nearby cars explained the LightVehicle suggestion, and one Truck confirmation
  where a visible cab/chassis contradicted the Container suggestion.
- Qwen3.6 35B-A3B no-prefix finalization validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_noprefix_focus_validation_actions_15_1780704776.json`
  reran the previous validation-failure rows plus clear action rows and
  completed 15/15 reviews with 0 backend failures, 0 final validation errors,
  and a single `finalize_review` schema pass for every row.
- Qwen3.6 35B-A3B no-prefix 100-crop validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_noprefix_wide100_100_1780705062.json`
  completed 100/100 reviews with 0 backend failures, 0 final validation errors,
  9 `accept_suggested`, 6 `confirm_current`, and 85 `skip_uncertain`. Compared
  with `vanch007_qwen36_local_consensus_textguard3_wide100_100_1780701940.json`,
  the same 100 rows dropped from 18 final-validation errors to 0 and from mixed
  retry sequences to one `finalize_review` pass for every row. Manual visual
  audit of `vanch007_qwen36_noprefix_wide100_100_1780705062_visual_non_skip.jpg`
  found no unsafe accepted class changes; the old UPole to Gastank accept became
  a safer skip because the visible target was a thin pole/bar overlapping a tank
  object.
- Qwen3.6 35B-A3B adjacent-text-guard focused benchmark:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_adjacent_text_guard_focus_15_1780707457.json`
  completed 15/15 reviews with 0 backend failures and 0 final validation
  errors, but visual audit exposed two unsafe overlap-driven accepts: a
  Boat/LightVehicle case where the target was a boat next to a car, and a
  UPole/Gastank case where the target was too entangled to relabel safely.
  This run motivated the current stricter partial-overlap rule.
- Qwen3.6 35B-A3B overlap-rebuttal focused benchmark:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_overlap_text_rebuttal_focus_15_1780707871.json`
  reran the same 15 rows after requiring explicit textual rebuttal for
  `partial_contamination` accepts. Result: 15/15 completed, 0 backend failures,
  0 final validation errors, 6 `accept_suggested`, 3 `confirm_current`, and 6
  `skip_uncertain`. The two unsafe overlap accepts were forced back to safe
  skips while the clean/rebutted non-skips remained defensible in the visual
  sheet.
- Qwen3.6 35B-A3B overlap-rebuttal 100-crop validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_overlap_text_rebuttal_wide100_100_1780708144.json`
  completed 100/100 reviews with 0 backend failures, 0 final validation errors,
  and one `finalize_review` schema pass for every row. Decisions were 8
  `accept_suggested`, 6 `confirm_current`, and 86 `skip_uncertain`. The
  anomaly scan found no non-skip decisions on limited/poor targets, no accepts
  with strong current evidence, and no bad-overlap accepts. Manual visual audit
  of
  `vanch007_qwen36_overlap_text_rebuttal_wide100_100_1780708144_visual_non_skip.jpg`
  found the 14 non-skip recommendations defensible; compared with the no-prefix
  100-crop run, UPole to LightVehicle and Container to Building overlap-heavy
  accepts moved to conservative skips, while the clear Truck to LightVehicle
  pickup case moved from skip to accept.
- Repeatable audit for that run:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_overlap_text_rebuttal_wide100_100_1780708144_audit.json`
  reports 100 records, 14 actionable recommendations, 0 unsafe issues, and 5
  guarded clear-target candidates.
- Self-contradictory accept reconciliation replay:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_overlap_text_rebuttal_wide100_100_1780708144_replayed_self_contradictory_confirm.json`
  re-expanded the same 100 compact outputs after adding a controller rule for
  cases where Qwen emitted `accept_suggested` while its own `target_class` and
  visible-fact rationale supported the current class. Result: 8
  `accept_suggested`, 9 `confirm_current`, and 83 `skip_uncertain`; the audit
  reports 0 unsafe issues and leaves the two risky partial-overlap cases guarded.
- Live Qwen3.6 35B-A3B self-contradiction 100-crop validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_selfcontradict_livewide100_100_1780731515_controller_fields.json`
  completed 100/100 reviews with 0 backend failures, 0 final validation errors,
  one `finalize_review` schema pass for every row, and decisions of 6
  `accept_suggested`, 7 `confirm_current`, and 87 `skip_uncertain`. The matching
  audit
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_selfcontradict_livewide100_100_1780731515_controller_fields_audit.json`
  reports 13 actionable recommendations and 0 unsafe issues. Manual visual
  audit found all 13 non-skip recommendations defensible, but also found that
  several guarded partial-overlap skips looked visually relabelable by a human.
  This confirms the current rail is precise but still conservative.
- Live Qwen3.6 35B-A3B overlap-recall 100-crop validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_overlap_recall_livewide100_100_1780734902.json`
  reran the same 100 records after broadening explicit overlap-rebuttal text
  detection for minor/adjacent overlap cases. Result: 100/100 completed, 0
  backend failures, 0 final validation errors, one `finalize_review` schema
  pass for every row, 11 `accept_suggested`, 7 `confirm_current`, and 82
  `skip_uncertain`. The matching audit
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_overlap_recall_livewide100_100_1780734902_audit.json`
  reports 18 actionable recommendations, 0 unsafe issues, and one guarded
  clear-target candidate. Compared with the prior live 100-crop run, exactly
  five rows moved from guarded skips to accepted class changes:
  Gastank->Building, Truck->LightVehicle, LightVehicle->UPole, and two
  Container->Building cases. Manual inspection of
  `vanch007_qwen36_overlap_recall_livewide100_100_1780734902_visual_all.jpg`
  and `vanch007_qwen36_overlap_recall_livewide100_100_1780734902_visual_non_skip.jpg`
  found all 18 non-skip actions defensible. The remaining visible issue is
  still conservative recall: guarded limited-quality or materially entangled
  candidates stay manual rather than being auto-recommended.
- Local-consensus routed probe:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_local_consensus_probe30_30_1780737074.json`
  reran the first 30 records with `enable_local_consensus_context=true`. Result:
  30/30 completed, 0 backend failures, 0 final validation errors, 4 routed
  `inspect_local_consensus_context` evidence packs, and decisions of 3
  `confirm_current` and 27 `skip_uncertain`. Compared with the 100-crop
  baseline subset, local consensus was safe but did not materially improve
  decision recall; keep it opt-in and policy-routed rather than default.
- Guarded-recommendation signal probe:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_guarded_signal_probe15_15_1780738681.json`
  reran the first 15 records after preserving guardrail-blocked model decisions
  as structured advisory data. Result: 15/15 completed, 0 backend failures, 0
  final validation errors, 0 unsafe audit issues, 1 actionable
  `confirm_current`, and 1 `guarded_recommendation`. The guarded case was the
  known UPole/Gastank partial-overlap row: the final decision stayed
  `skip_uncertain`, but the benchmark/UI now expose the blocked
  `accept_suggested -> Gastank` signal and the exact overlap guardrail reason
  for human triage.
- Guarded-recommendation 100-crop validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/vanch007_qwen36_guarded_signal_wide100_100_1780739423.json`
  reran the full 100-record benchmark with the guarded-signal branch. Result:
  100/100 completed, 0 backend failures, 0 final validation errors, 0 unsafe
  audit issues, one `finalize_review` schema pass for every row, 11
  `accept_suggested`, 7 `confirm_current`, 82 `skip_uncertain`, and 3
  `guarded_recommendation` records. The comparison against
  `vanch007_qwen36_overlap_recall_livewide100_100_1780734902.json` reports 100
  matched rows and 0 decision/target/confidence drift. The guarded rows are
  UPole->Gastank blocked by material partial overlap, Boat->LightVehicle blocked
  by limited backend visual quality, and Container->Building blocked by limited
  backend visual quality. The generated
  `vanch007_qwen36_guarded_signal_wide100_100_1780739423_visual_guarded.jpg`
  contact sheet makes those blocked-but-plausible human-review cases visible
  without weakening automatic relabel rails.
- Class-concept-brief action-slice validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/concept_brief_v2_actions_refixed_33_41_9_1780745806.json`
  reran the action-heavy rows 33-41 from the guarded-signal 100-crop benchmark
  with cached visual concept briefs enabled. Result: 9/9 completed, 0 backend
  failures, 0 final validation errors, 0 unsafe audit issues, 5
  `accept_suggested`, 2 `confirm_current`, 2 `skip_uncertain`, and 0 guarded
  recommendations. The run recovered useful class-changing recommendations that
  had previously been over-blocked by too-narrow text guards, including
  UPole->LightVehicle where the overlap was background road markings rather
  than a vehicle, and Truck->LightVehicle where the target was a pickup-like
  light vehicle.
- Class-concept-brief 30-crop safety probe:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/concept_brief_v2_probe30_refixed_30_1780746210.json`
  reran the first 30 guarded-signal rows with concept briefs enabled. Result:
  30/30 completed, 0 backend failures, 0 final validation errors, 0 unsafe
  audit issues, 3 `confirm_current`, 27 `skip_uncertain`, and 0 guarded
  recommendations. The run had only 4 clear backend-tier targets, so the low
  action rate is expected; the three non-skip confirmations were Boat,
  Building, and Truck cases where overlap or neighbor similarity did not justify
  changing the current class. The run generated 2 missing concept briefs and
  used 58 concept-brief cache hits.
- Pair-contrast-brief 30-crop safety probe:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/pair_contrast_v1_probe30_guarded_confirm_30_1780749413.json`
  reran the same 30 guarded-signal rows with per-pair contrast briefs. Result:
  30/30 completed, 0 backend failures, 0 final validation errors, 0 unsafe audit
  issues, 3 `confirm_current`, 27 `skip_uncertain`, and 0 guarded
  recommendations. The run generated 9 missing pair-contrast briefs, used 21
  pair cache hits, and visually checked the three actionable confirmations:
  Boat->LightVehicle was a small boat or boat-like object, Building->LightVehicle
  was mostly roof/building pixels with vehicle contamination at the edge, and
  Truck->Building remained a weak but acceptable truck-like edge case rather
  than an automatic class change.
- Pair-contrast hardening note:
  the first pair-contrast artifacts exposed that top-anchor-only exemplars could
  make class concepts too narrow, for example Boat summarized as "large vessels"
  despite small boats appearing in the review set. Version
  `class_visual_concept_brief_v3` / `class_pair_contrast_brief_v2` invalidates
  those caches and uses trusted-diverse projection sampling plus stricter
  `must_skip_when` normalization.
- Trusted-diverse concept/pair-contrast action-slice validation:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/pair_contrast_v2_diverse_actions_33_41_9_1780751361.json`
  reran the same action-heavy nine rows after the v3/v2 cache bump. Result: 9/9
  completed, 0 backend failures, 0 final validation errors, 0 unsafe audit
  issues, 4 `accept_suggested`, 2 `confirm_current`, 3 `skip_uncertain`, and 1
  guarded recommendation. Compared with the prior pair-contrast run, one local
  benchmark row improved from conservative skip to advisory class-change after
  visual inspection showed target-specific cues, while a contradictory
  confirmation stayed blocked by generic evidence conflict handling.
- Trusted-diverse 30-crop safety probe and manual guard follow-up:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/pair_contrast_v2_diverse_probe30_30_1780751915.json`
  completed 30/30 with 0 backend failures and 0 final validation errors, but
  manual visual audit found one accepted class change too aggressive: the target
  was partially contaminated by nearby context and did not have enough
  independent suggested-class support for an automatic recommendation. The
  follow-up fix is generic: partial-contamination class changes now require
  strong suggested anchors, strong local context, strong global context, and an
  explicit rationale that the overlap does not explain the target features.
- Target-only final-vision hardening:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/grounded_evidence_v16_subprocess30_30_1780772779.json`
  initially completed 30/30 under subprocess isolation with 0 failures, but the
  updated audit
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/grounded_evidence_v16_subprocess30_30_1780772779_audit_after_anchor.json`
  correctly flags one accepted class change as unsafe because
  `anchor_evidence_suggested` was only moderate. A targeted rerun after removing
  class-context-pack images from final vision,
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/grounded_evidence_v18_item26_target_only_final_1_1780775482.json`,
  forced that same row to `skip_uncertain` and preserved the attempted
  `accept_suggested` as a guarded recommendation.
- Target-only final-vision 30-crop safety probe:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/grounded_evidence_v19_target_only_final_subprocess30_30_1780775543.json`
  completed 30/30 under per-review subprocess isolation with 0 failures, 0 final
  validation errors, and 0 unsafe audit issues. Decisions were 30
  `skip_uncertain`; four clear-tier rows preserved blocked model class-change
  attempts as `guarded_recommendation`. This is intentionally conservative and
  useful for human triage, but not yet a high-recall automatic relabeler.
- Benchmark harness note: aggregate benchmark records must persist the
  controller-normalized evidence fields (`current_evidence`,
  `suggested_evidence`, `target_evidence`, `overlap_assessment`), the explicit
  `visible_target_cues` ledger, `supporting_clean_evidence_ids`, the controller
  `evidence_ledger`, the nested `cue_verifier` result, and
  `model_compact_arguments`. Otherwise the audit can miss missing-cue,
  current-class-plausibility, or ungrounded class changes, or incorrectly fall
  back to stale compact model fields after a controller reconciliation.
- Deterministic scale/embedding context focused check:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/deterministic_tools_text_only_guardrail_check_4_1780781763.json`
  completed 4/4 with 0 unsafe audit issues. It produced 2 actionable
  suggestions and 2 guarded skips. The final structured
  `same_image_*_evidence` fields matched the controller report signals. Report
  artifacts were saved for audit, but their text-card images were not sent to
  Qwen as visual inputs.
- Deterministic scale/embedding context 30-crop benchmark:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/deterministic_tools_text_only_v1_30_1780782042.json`
  completed 30/30 with 0 failures, 0 final validation errors, and 0 unsafe
  audit issues. Decisions were 2 `accept_suggested` and 28 `skip_uncertain`.
  There were 2 guarded model class-change attempts. Scale signals were 10
  `supports_current`, 7 `questions_current`, 4 `neutral`, and 9
  `insufficient`; embedding signals were 9 `supports_current`, 4
  `questions_current`, 8 `neutral`, and 9 `insufficient`. The reports improved
  grounding and auditability and changed two clear-tier rows from guarded skips
  to actionable suggestions. Manual visual inspection found those two suggestions
  plausible, but they should remain advisory until a larger labeled benchmark
  proves that loosening relabel rails improves precision.
- Dominant-current-overlap 100-crop release gate:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/deterministic_tools_dominant_overlap_guard_deep100_100_1780785210.json`
  replayed the guarded-signal 100-record set with local consensus, concept
  briefs, pair contrasts, deterministic scale/embedding reports, and the new
  dominant-current-overlap guard. Result: 100/100 completed, 0 backend failures,
  0 final validation errors, 0 unsafe audit issues, 13 `accept_suggested`, 87
  `skip_uncertain`, and 6 guarded model recommendations. The audit path is
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/deterministic_tools_dominant_overlap_guard_deep100_100_1780785210_audit.json`.
  The previously unsafe Building-to-LightVehicle case is now guarded because
  building pixels dominate the target bbox while the candidate vehicle evidence
  comes from overlap/context. Manual contact-sheet inspection found the remaining
  actionable rows plausible but still advisory.
- Limited-advisory 100-crop activity gate:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/limited_advisory_deep100_100_1780789010.json`
  reran the same 100 records after enabling limited-quality final review and
  changing the limited-tier final instruction from "class-changing decisions are
  forbidden" to "give your best human-triage opinion." Result: 100/100
  completed, 0 backend failures, 0 final validation errors, 0 unsafe audit
  issues, 13 `accept_suggested`, 87 `skip_uncertain`, and 44 guarded model
  recommendations. The audit path is
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/limited_advisory_deep100_100_1780789010_audit.json`.
  Compared with `deterministic_tools_dominant_overlap_guard_deep100`, final
  Qwen calls increased from 19 to 70 because limited targets now reach the
  final state; guarded recommendations increased from 6 to 44. The model was
  therefore not inherently unopinionated: on limited targets it emitted 38
  `accept_suggested` opinions and 13 `skip_uncertain` opinions. All limited
  non-skip opinions remain non-mutating `guarded_recommendation` records because
  the backend still requires clear target quality for actionable class changes.
  Visual inspection of the actionable contact sheet found the 13 clear-tier
  accepts broadly plausible; `limited_advisory_deep100_100_1780789010_visual_guarded.jpg`
  makes the 44 guarded opinions visible for manual audit. The limited-tier
  suggestions remain useful for human triage but too visually weak for automatic
  relabeling.
- Moderate-anchor clear-target experiment:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/moderate_anchor_clear_relabel_deep100_100_1780791467.json`
  tested a narrow recall increase for clear targets whose suggested anchors were
  only moderate but whose target, local context, global context, and overlap
  fields otherwise supported the suggested class. It completed 100/100 with 0
  backend failures and 44 guarded recommendations, but visual review found that
  one Boat-to-LightVehicle accept relied on negative/context cues rather than
  positive target-object evidence. This experiment was rejected as the final
  gate and motivated stricter cue normalization.
- Positive-visible-cue 100-crop gate:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/positive_cue_guard_deep100_100_1780792746.json`
  reran the same 100 records after stripping negative, absence, color-only, and
  context-only cue phrases from the class-change cue ledger. Result: 100/100
  completed, 0 backend failures, 0 final validation errors, 0 unsafe audit
  issues, 8 `accept_suggested`, 92 `skip_uncertain`, and 45 guarded model
  recommendations. The audit path is
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/positive_cue_guard_deep100_100_1780792746_audit.json`.
  Visual inspection of
  `positive_cue_guard_deep100_100_1780792746_visual_non_skip.jpg` found the
  remaining actionable changes structurally plausible, while the rejected weak
  vehicle/boat/context cases stayed visible as guarded human-review suggestions.
  This is the current preferred balance: fewer automatic recommendations than
  the limited-advisory run, but clearer distinction between safe actions and
  useful non-mutating model opinions.
- Guardrail bottleneck follow-up:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/semicolon_repair100_100_1780799314.json`
  reran the same 100 records after two targeted fixes. First, partial
  `verify_visible_cues` outputs now get one repair turn so malformed verifier
  JSON is not mistaken for an evidence failure. Second, clear targets with only
  one concrete visible cue can accept the suggested class only when independent
  support is strong: clear backend/visual/object visibility, weak current
  evidence, strong suggested/target/anchor/local/global evidence,
  `local_consensus_evidence=supports_suggested`, no overlap contamination, and
  deterministic same-image scale/embedding signals that do not support the
  current class. This path is capped at confidence `0.86` and records an
  advisory reason. A follow-up audit also tightened semantic text-conflict
  parsing so `no cargo; matches LightVehicle visual cues` is not treated as
  rejecting the LightVehicle target class. Result: 100/100 completed, 0 backend
  failures, 0 unsafe audit issues, 11 `accept_suggested`, 89 `skip_uncertain`,
  and 50 guarded model recommendations. The audit path is
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/semicolon_repair100_100_1780799314_audit.json`.
  This supports the diagnosis that the model is often opinionated enough; the
  remaining low action rate is mainly caused by explicit guardrails and weak
  target evidence. Further recall increases should improve rendered evidence or
  add non-mutating human-triage surfacing before relaxing mutation gates.
- MLX finalization stability recovery:
  earlier local probes after final-context compaction hit Metal command-buffer
  timeouts on the MLX visual finalizer, including reduced tests with a
  two-message final state, 384px image cap, one clean target image, the 2B MLX
  checkpoint, and explicit runtime resets. The current Mac policy is the
  opposite of a controller-only fallback: `CLASS_ANALYSIS_QWEN_REVIEW_ENABLE_MLX_FINAL`
  defaults to `true`, the model-facing schema uses `final_class`, and test runs
  use a larger final/verifier token budget so the local VLM can produce complete
  reasoning. Disable MLX finalization only as an explicit fallback/debug mode.
- Stabilized controller 30-crop replay:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/stabilized_mixed30_tightened_30_1780815007.json`
  replayed the same 30 rows as
  `stabilized_mixed30_30_1780814217.json` with MLX finalization disabled. Result:
  30/30 completed, 0 backend failures, 0 unsafe audit issues, 1
  `confirm_current`, 29 `skip_uncertain`, and 1 guarded human-triage class-change
  hint. Visual inspection of the prior guarded sheet showed two local-consensus
  only hints were not reliable enough; the tightened path now requires local
  consensus plus scale or embedding evidence that questions the current class.
  The remaining guarded hint is Truck-to-Building with same-image consensus,
  scale, and embedding all questioning the current class. This was the
  then-current controller-only fallback balance while MLX finalization was
  unstable; it is preserved here as historical evidence, not as the current Mac
  default.

The validation proves the backend/tool protocol and deterministic evidence
packaging are stable enough for VLM-centered review experiments. Qwen decisions
remain human-controlled and rails may block automatic mutation, but the VLM
final judgment is the product core: deterministic overlap, scale, embedding,
quality, and cue checks must preserve the raw model recommendation and its
reasoning as audit context. The current strict-overlap and plausibility behavior
is intentionally conservative for mutation, not for triage: guarded class-change
opinions are still useful human-review signals, while automatic relabel advice
is allowed only when the target is clear and the current class is not plausibly
supported by clean target/source evidence.

The current controller implementation adds glossary/guidance injection, overlap
decomposition, the class context pack, deterministic required-evidence
rendering, mandatory clean zoom context, and trusted-diverse concept/pair
briefs. Regression tests cover:

- final-schema validation and non-mutation
- quality-gate forced skip behavior
- class-change blocking on limited visual quality
- class-change blocking on material overlap and dataset-derived subtype
  ambiguity
- class-change blocking when current-class material overlap dominates the target
  bbox and the suggested class is only weakly present
- self-conflicting accept/confirm blocking when the model reports strong
  evidence for the competing class
- skip-confidence capping
- overlap-contamination forced skip behavior
- required-tool loop enforcement and evidence artifact writes
- controller evidence-ledger artifact, event, final-result summary, and final
  prompt injection
- per-turn model input/output JSONL logging
- clean source image emission from the source-overlay tool
- local consensus clean/dot evidence rendering
- mandatory clean zoom evidence before finalization
- local consensus final-schema and guardrail enforcement
- deterministic same-image scale report generation, final-schema normalization,
  and benchmark export counters
- deterministic same-image embedding report generation from existing analysis
  vectors, final-schema normalization, and benchmark export counters
- partial-overlap decomposition
- trusted-anchor selection
- glossary/guidance prompt inclusion
- cached class concept and pairwise contrast brief generation, image artifact
  writes, cache reuse, and final-prompt injection as advisory memory
- visible target cue ledger normalization and class-change blocking when cues are
  absent or class-label-only
- target-detail and poor-advisory benchmark:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/poor_advisory_subproc30_30_1780805865.json`
  completed 30/30 reviews with subprocess isolation, 0 backend failures, 0
  final validation errors, and 0 unsafe audit issues. It produced 2 actionable
  class changes and 15 guarded human-triage opinions. Manual visual audit found
  the two actionable changes defensible and showed that limited-quality guarded
  opinions were useful, while poor-quality advisory added little and could be
  noisy. The app default therefore keeps limited advisory enabled but skips poor
  targets unless `allow_poor_final_review=true` is explicitly requested.
- genericity/verifier replay check on the current wide-100 artifact:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/sddf_dual_v3_wide100_100_1780857344.json`
  recorded one cue-verifier call, but the current verifier gate would route ten
  clear/reviewable guarded recommendations (`6, 13, 26, 33, 34, 39, 40, 41, 66,
  87`) into the second VLM pass. Those rows are exactly the cases we should use
  for the next recall benchmark: clear target pixels, weak current evidence,
  strong target/suggested evidence, target-specific cues, and a guardrail that
  needs VLM-grounded cue verification rather than a passive skip. The 30 poor
  backend-tier rows remain expected no-signal rows unless better source evidence
  is rendered.
- genericity v5 review-fix benchmark:
  `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/genericity_v5_reviewfix_wide100_100_1780878216.json`
  completed 100/100 reviews with 0 failures, 0 final validation errors, and 0
  unsafe audit issues. Compared with
  `genericity_v4_wide100_100_1780875777.json`, the corrected verifier-backed
  partial-overlap path changed one row: `#39 LightVehicle->UPole` moved from a
  guarded skip to `accept_suggested` after the verifier supplied target-specific
  visible cues plus an overlap rebuttal and same-image embedding evidence that
  questioned the current class. The run produced 5 actionable recommendations
  instead of 4, with 65 guarded recommendations, 70 effective human signals, and
  2 cue-verifier promotions. Visual inspection of the non-skip sheet found the
  five actionable recommendations plausible. The corresponding code audit also
  removed benchmark-side concrete object-token whitelists; visible-cue
  filtering must remain dataset-agnostic and rely on the labelmap, glossary,
  generated concept briefs, and actual model evidence rather than committed
  project vocabulary.

A larger labeled real-model benchmark should be run before treating v2
recommendations as more than advisory.

## Provenance References

The local code intentionally keeps only the parts that are useful for label
review:

- Qwen-Agent README: a framework for Qwen instruction following, tool usage,
  planning, and memory; see https://github.com/QwenLM/Qwen-Agent.
- Qwen function-calling guidance: model calls are constrained by explicit
  function/tool descriptions and JSON arguments; see
  https://qwen.readthedocs.io/en/v2.0/framework/function_call.html.
- Qwen3-VL Think with Images cookbook: demonstrates iterative visual inspection
  with an `image_zoom_in_tool`; see
  https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/think_with_images.ipynb.
- Hermes Function Calling prompt assets: one-function-at-a-time JSON tool calls;
  see https://github.com/NousResearch/Hermes-Function-Calling.
- OpenAI practical agent guidance: keep tools, guardrails, and human oversight
  explicit around the agent loop; see
  https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/.
- OpenClaw agent-loop/context/loop-detection docs: bounded observe/act loops,
  context accumulation, and loop detection; see https://docs.openclaw.ai.
- VisHarness multi-turn visual expert routing: evidence-gathering visual agent
  framing with heterogeneous visual context before a decision; see
  https://arxiv.org/html/2605.29894v1.
- Flamingo visual in-context learning: few-shot multimodal exemplars motivate
  the clean trusted-example concept brief layer; see
  https://arxiv.org/abs/2204.14198.
- Visual In-Context Learning for Large Vision-Language Models: retrieved visual
  demonstrations, task-oriented summarization, and compact composition motivate
  caching per-class concept briefs instead of sending unbounded examples; see
  https://arxiv.org/abs/2402.11574.
- VISCO visual self-correction: visual critiques can help when they force the
  model to look back at image evidence, but naive self-critique can be
  detrimental; see https://arxiv.org/abs/2412.02172.

When changing this flow later, preserve the same invariants unless there is a
specific test-backed reason to change them: bounded turns, evidence-first final
decision, schema validation, no automatic label mutation, artifact logging,
review-specific tools only, and a fresh benchmark audit. When adding a new tool,
update the prompt, backend required-tool policy, frontend display, tests, and
this document in the same change.
