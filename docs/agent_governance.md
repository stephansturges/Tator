# Agent Governance And Anti-Drift Rules

This document explains how future agents should change Tator without drifting
away from the user's intended product goal. The short executable contract is
[AGENTS.md](../AGENTS.md). Durable project memory is [memory.md](../memory.md).

## Why This Exists

The Class Split Qwen review work exposed a dangerous failure mode: the
implementation drifted from a VLM-centered review workflow into a conservative
controller fallback that was safer but not the product the user asked for. That
burned time and tokens while reducing the value of the tool.

The prevention mechanism is simple:

- name the intended product behavior before changing architecture
- identify the core mechanism that must be preserved
- keep safety rails around that mechanism instead of replacing it
- benchmark usefulness as well as safety
- document provenance when importing external agent patterns

## Source-Guided Practices

These rules are grounded in current public guidance and research:

- OpenAI's practical agent guidance emphasizes explicit orchestration,
  guardrails, evaluation, and human oversight for agent systems:
  https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
- Anthropic's Claude Code memory guidance recommends persistent project memory
  files for repository-specific instructions:
  https://docs.anthropic.com/en/docs/claude-code/memory
- ReAct frames tool-using agents as interleaved reasoning and acting, which is
  the pattern we want for evidence-gathering review loops:
  https://arxiv.org/abs/2210.03629

Use these as provenance anchors, not as vague decoration. If a future change
imports a new agent-loop, prompt, eval, or guardrail pattern, cite the exact
source in the relevant local doc.

## Required Change Discipline

For substantial feature work, especially agentic or ML-backed flows, every
change must answer these questions in the doc or final summary:

1. What is the user story?
2. What is the core mechanism that must stay intact?
3. What safety rails are being added, and what are they not allowed to replace?
4. What evidence proves the mechanism still runs?
5. What evidence proves the outputs are useful, not merely safe?

If those questions cannot be answered, the work is not ready to be called done.

## Guardrails Versus Product Behavior

Guardrails are constraints on a product behavior. They are not substitutes for
that behavior.

Allowed:

- block automatic mutation when evidence quality is poor
- preserve a model recommendation as guarded human-triage output
- require citations to clean evidence images before accepting a class change
- keep raw model input/output logs for audit
- expose explicit fallback feature flags for broken runtimes

Not allowed:

- disable the requested model reasoning path by default because it is unstable
- replace a VLM decision with deterministic preflight logic and call it an agent
- optimize for zero unsafe actions by making the workflow skip everything
- hide model recommendations that were blocked by rails
- remove runtime cost before proving the product loop works

## Benchmark Standard For Agentic Visual Review

Safety-only metrics are insufficient. Benchmark artifacts must include:

- number of completed reviews
- backend failures and validation failures
- raw model input/output logs or paths
- final decisions
- actionable recommendations
- guarded recommendations
- useful negative confirmations
- unsafe actions
- manual visual audit notes for a representative sample
- a short interpretation of whether failures are model limitations, prompt
  limitations, evidence-pack limitations, runtime limitations, or guardrail
  overreach

If a benchmark has zero or near-zero useful outputs, the next action is to debug
model/runtime/context/prompt/rails. Do not redefine that as success.

## Class Split Qwen Review Invariant

The Qwen likely-wrong review workflow exists to let a local VLM inspect target
detail, wider source context, overlap decomposition, trusted class examples,
class glossary/guidance, local consensus, scale reports, and embedding reports,
then produce its own advisory decision.

The controller owns evidence rendering, validation, and mutation safety. It does
not own the visual decision by default.

Any change that makes deterministic controller logic the normal final decision
path must be treated as a product regression unless the user explicitly asks for
that mode.

## Documentation Standard

When changing agentic workflows:

- update the workflow-specific doc in the same patch
- include source links for external methods
- record benchmark commands and artifact paths
- note known runtime/model caveats
- keep stale benchmark history clearly separated from current policy

Do not rely on chat history for these facts. Put them in repository files.
