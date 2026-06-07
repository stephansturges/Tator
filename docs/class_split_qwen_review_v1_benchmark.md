# Class Split Qwen Review V1 Benchmark

This document canonicalizes the first full V1 benchmark for the Class Split
Qwen likely-wrong review workflow. The benchmark is intentionally slow and
VLM-centered: the point is to evaluate whether the local Qwen reviewer produces
useful visual judgment, not to replace it with a deterministic controller.

## External Agent References

The V1 interpretation uses three external references:

- OpenAI, "A practical guide to building agents":
  https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
  The relevant standard for this workflow is explicit tools, structured
  instructions, guardrails, and human handoff for high-risk actions.
- Anthropic, "Building effective agents":
  https://www.anthropic.com/engineering/building-effective-agents
  The relevant standard is simple composable patterns, transparent tool
  interfaces, and evaluating whether extra agent complexity improves outcomes.
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models":
  https://arxiv.org/abs/2210.03629
  The relevant standard is interleaving reasoning with environment observations
  so the model updates its plan from tool evidence instead of guessing from one
  static prompt.

For this repo, those references translate to a concrete invariant: deterministic
checks can gate risky label-changing advice, but they must remain rails around a
real VLM judgment. They are not a substitute for the VLM.

## UI Workflow

1. Open the current annotation dataset in `Label Images`.
2. Open `Class Split Explorer`.
3. Run an all-class Class Split analysis so likely-wrong candidates are
   available.
4. Review the `Likely wrong class` vignette strip below the graph.
5. Optional: open `Qwen review context` and edit:
   - `labelmap_glossary`: class definitions for the current dataset.
   - `review_guidance`: temporary session instructions for edge cases or class
     policy.
6. Select a Qwen reviewer model in the vignette toolbar.
7. Click `Review with Qwen` on one vignette.
8. Read the advisory result:
   - `accept_suggested`: Qwen recommends changing to the suggested class.
   - `change_to_other`: Qwen recommends a different explicit class.
   - `confirm_current`: Qwen thinks the current class is probably correct.
   - `skip_uncertain`: Qwen or the backend cannot support a confident action.
   - `guarded`: Qwen produced a potentially useful opinion, but backend rails
     blocked automatic label-change advice and surfaced it for human triage.
9. Apply the actual label edit manually with `Confirm current class`,
   `Switch class to ...`, `Skip`, or `See instance`.

Qwen never mutates labels directly in V1. This keeps the workflow aligned with
agent safety guidance for high-risk or irreversible actions while preserving the
model's visual opinion for human review.

## Benchmark Command

The canonical V1 run used the completed all-class job
`ca_c5c4a7d6ea` and the previous wide 100-row controller run as source rows:

```bash
.venv-macos/bin/python tools/run_class_split_qwen_review_benchmark.py \
  --job-id ca_c5c4a7d6ea \
  --source-run uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/controller_preflight_wide100_100_1780816362.json \
  --count 100 \
  --run-label v1_vlm_finalclass_wide100 \
  --model-id vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit \
  --enable-local-consensus \
  --enable-class-concept-briefs \
  --mlx-reset-every 0 \
  --visual-limit 100 \
  --audit \
  --compare-run uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/controller_preflight_wide100_100_1780816362.json
```

The run intentionally reuses the MLX model in-process. A previous cold-start
subprocess attempt was aborted because it reloaded the 35B checkpoint for every
row and was not a realistic UI-path benchmark.

## Run Artifacts

Relative to the repo root:

- `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/v1_vlm_finalclass_wide100_100_1780824743.json`
- `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/v1_vlm_finalclass_wide100_100_1780824743_audit.json`
- `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/v1_vlm_finalclass_wide100_100_1780824743_visual_non_skip.jpg`
- `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/v1_vlm_finalclass_wide100_100_1780824743_visual_guarded.jpg`
- `uploads/class_analysis/ca_c5c4a7d6ea/qwen_reviews/v1_vlm_finalclass_wide100_100_1780824743_visual_all.jpg`

These artifacts are local benchmark evidence. They are not committed because
they are generated run outputs.

## Quantitative Result

- Run id: `v1_vlm_finalclass_wide100_100_1780824743`
- Model: `vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit`
- Sample size: 100 vignettes
- Completed: 100
- Failed: 0
- Elapsed: 13,332 seconds, approximately 3 hours 42 minutes
- Backend visual tiers: 19 clear, 51 limited, 30 poor
- Final decisions: 3 `accept_suggested`, 97 `skip_uncertain`
- Effective human signals: 70
- Guarded human triage signals: 66
- Guarded model recommendations: 67
- Unsafe audit issues: 0
- Final validation errors: 0
- VLM finalization rows: 70
- Poor-target rows skipped before VLM finalization: 30

Schema sequence counts:

- Empty sequence: 30
- `finalize_review`: 66
- `concept_pair_contrast->finalize_review`: 2
- `concept_brief->concept_pair_contrast->finalize_review`: 2

Review disposition counts:

- `actionable_class_change`: 3
- `guarded_overlap_risk`: 15
- `guarded_visual_quality`: 51
- `target_not_reviewable`: 30
- `verified_current_class_overlap`: 1

Compared with the previous controller-only wide run, V1 produced real final Qwen
schema calls for the 70 non-poor rows and preserved model opinions as guarded
human-triage signals instead of returning blank controller-only outputs.

## Manual Expected-Output Audit

The non-skip sheet contained three actionable class-change recommendations.
Manual visual audit found two likely correct recommendations and one likely
false positive.

| Row | Agent result | Manual expectation | Assessment |
| --- | --- | --- | --- |
| 66 | `Truck -> LightVehicle`, confidence 0.72 | Keep current `Truck` or at least do not accept as `LightVehicle` | Likely false positive. The target crop shows a white box truck or van-like vehicle. Qwen called it a sedan. |
| 83 | `Truck -> Building`, confidence 0.72 | Change to `Building` | Likely correct. The target is a fixed white rectangular roof or structure in a paved area, not a truck. |
| 92 | `Building -> Solarpanels`, confidence 0.72 | Change to `Solarpanels` | Likely correct. The target shows a panel grid distinct from surrounding roof/building texture. |

Actionable precision on the manual actionable subset is therefore about 2/3.
That is not enough for automatic relabeling, but it is useful enough to justify
continued VLM-centered development.

The guarded clear-target sheet showed that several blocked model opinions were
visually plausible. Examples include cases where Qwen saw vehicles, poles,
boats, or structures inside overlapping or contaminated boxes. This is useful
signal, not noise. The current V1 rails are deliberately conservative, but the
future goal is to convert more guarded opinions into high-quality human triage
or safe automation without accepting weak vehicle subtype errors like row 66.

## Critical Review

V1 is a real step back toward the intended product because the VLM is again
making final visual judgments on every non-poor target. It is not yet good
enough to trust as an automatic label changer.

The strongest positives:

- The run completed 100/100 with no schema failures.
- Poor targets were correctly excluded from final VLM calls, avoiding waste.
- Limited targets still reached Qwen finalization, which produced many guarded
  but useful human-review signals.
- The benchmark now logs prompt inputs, model outputs, evidence images,
  deterministic rails, and audit sheets, making failures inspectable.
- Concept brief and pair contrast calls executed and cached instead of being
  theoretical features.

The strongest failures:

- The action rate is only 3/100. The model is visually active, but the system is
  still too conservative to materially accelerate review on its own.
- Manual audit found one bad actionable class change out of three. That means
  the audit invariants do not yet catch every semantic error.
- Vehicle subtype reasoning remains weak. The row 66 false positive shows that
  Qwen can call a box truck or van-like target a sedan when the crop is clean.
- The guarded recommendation count is high. This indicates the VLM is producing
  useful opinions that the controller cannot yet reconcile into a clean user
  decision.
- Runtime is long. The 35B MLX path is acceptable for deep benchmarking, but
  not for routine unattended review unless we batch, cache, or add model tiers.

## V2 Improvement Directions

The next jump should not be another small prompt tweak. Based on the benchmark
logs and the external agent references, the useful direction is a stronger
ReAct-style visual evidence loop with explicit observations:

1. Split finalization into target identity, overlap decomposition, and class
   decision substeps. The current final JSON asks Qwen to do too much in one
   output.
2. Make guarded recommendations first-class UI cards rather than hidden
   `skip_uncertain` text. Humans should see that Qwen had a concrete opinion and
   why it was blocked.
3. Add a vehicle-size and morphology discriminator that is generic, not
   dataset-specific: elongated box vehicle, passenger-car footprint, roof-only
   structure, narrow vertical object, panel grid, container-like rectangular
   object. These should be visual descriptors generated per dataset/class
   concept brief, not hard-coded labels.
4. Add a second-pass verifier only for proposed class changes. It should see
   the target crop, source context, clean same-class anchors, suggested-class
   anchors, and the first model's visible cues, then explicitly try to falsify
   the proposed change.
5. Introduce disagreement-based escalation. If first-pass Qwen and verifier
   disagree, surface a high-priority guarded card instead of collapsing to a
   generic skip.
6. Benchmark against manually inspected rows, not only schema/audit invariants.
   V1 proved that zero unsafe audit issues can still hide semantic false
   positives.
7. Keep automatic mutation disabled until manual precision is high on actionable
   rows and guarded rows can be reliably separated into "needs human" versus
   "safe suggestion".

This is the baseline to beat. Future benchmark reports should compare against
the V1 metrics above and include both automated audit counts and manual visual
precision on at least the actionable subset.
