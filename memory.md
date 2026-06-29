# Project Memory

This file records durable project decisions that future agents must preserve.
Operational instructions are in [AGENTS.md](AGENTS.md); detailed governance and
source references are in [docs/agent_governance.md](docs/agent_governance.md).

## 2026-06-07: Anti-Drift Rule For Agentic Product Work

The Class Split Qwen review workflow drifted from the intended product goal:
"let a local VLM inspect rich visual/context evidence and make its own advisory
decision before the human reviews a likely-wrong annotation" into a safer but
much weaker controller-dominated system that skipped or bypassed most final VLM
reasoning.

This must not happen again.

Durable rule:

- Preserve the user-requested core mechanism. For Qwen review, the VLM final
  judgment is the core; deterministic checks are rails, not replacements.
- If the core mechanism fails, debug the model/runtime/schema/context/harness or
  mark the task blocked. Do not silently redefine success as "no unsafe
  mutations."
- Guardrails may block automatic label changes, but they must preserve the raw
  model recommendation and expose it as a human-triage signal when useful.
- Benchmarks for agentic workflows must report both safety and usefulness:
  completed rows, validation errors, unsafe actions, actionable recommendations,
  guarded recommendations, human-triage signals, raw model I/O logs, and manual
  visual audit notes for visual tasks.
- Early experimental workflows should prioritize product truth over runtime
  cost. Optimize only after the intended loop is working.

For the current Qwen likely-wrong review work, see
[docs/class_split_qwen_review_agent.md](docs/class_split_qwen_review_agent.md).

## 2026-06-29: Caption Set-And-Forget Resume Invariants

Set-and-forget caption jobs must be able to save their own generated captions
while still blocking unrelated user/UI caption mutations during active backend
runs. Preserve the active job id bypass only for internal caption job writes.

Resume against an existing caption artifact directory must reuse that artifact's
manifest case set. Do not rebuild the resume case list from current text labels,
because newly saved captions can otherwise shrink the case list and fail runner
preflight against the original artifact manifest.

Hard launch failures such as preflight, pilot-certification, and backend
supervision failures are non-resumable until their underlying gate is fixed.
Interrupted wrapper jobs without runner/preflight evidence are also
non-resumable; stale browser tabs must receive a backend 409 instead of
spawning another wrapper job.
