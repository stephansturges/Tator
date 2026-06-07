# Agent Operating Instructions

This file is the repo-local operating contract for coding agents working on
Tator. Read it before changing code. Keep it short; durable product memory lives
in [memory.md](memory.md), and detailed rationale lives in
[docs/agent_governance.md](docs/agent_governance.md).

## Non-Negotiable Product Contract

- Tator is an annotation-assistance stack. Preserve user data, preserve the
  user's currently open dataset state, and keep final annotation changes
  human-controlled unless the user explicitly asks for automation.
- Do not replace a requested product capability with a weaker safety fallback
  and call the work done. If a core mechanism is failing, fix that mechanism,
  swap in a suitable implementation, or report the blocker clearly.
- For Class Split Qwen review, the local VLM final judgment is the product core.
  Deterministic overlap, scale, embedding, quality, and cue checks are rails and
  audit context. They may block automatic mutation, but they must not replace
  the VLM's reasoning by default.
- If a model/runtime path fails, prefer solving the runtime, model selection,
  prompt schema, context pack, or benchmark harness. A controller-only fallback
  is allowed only behind an explicit fallback setting or explicit user request.
- A "safe" benchmark is not sufficient when the product goal is decision help.
  Benchmark reports must include usefulness/action rate, guarded VLM signals,
  raw model inputs/outputs, and manual visual audit notes when the task is visual.

## Anti-Drift Workflow

Before substantial changes:

1. Restate the intended user story and identify the core mechanism that must not
   be removed.
2. Inspect the current implementation and existing docs before proposing a new
   architecture.
3. If changing prompts, agent loops, guardrails, model selection, or evidence
   packaging, update the relevant doc in the same change.
4. Add or update tests that fail if the core mechanism is bypassed.
5. Run an effectiveness check, not only a safety check, for agentic workflows.

During implementation:

- Preserve provenance: cite external patterns in docs when importing an
  agent-loop, prompt, eval, or guardrail practice.
- Treat guardrails as constraints around the requested behavior, not as a reason
  to remove the behavior.
- Keep deterministic controller decisions observable. If they override a model
  recommendation, preserve the raw model recommendation, override reason, and
  human-triage signal in artifacts and UI-facing payloads.
- Do not optimize away cost or runtime for early experimental workflows unless
  the user asks. First prove the intended product loop works; then optimize.

Before finalizing:

- Verify with code tests and, when relevant, benchmark artifacts.
- Say exactly what was verified and where the evidence is.
- If the intended core mechanism is still not working, do not present the task
  as complete. Explain the blocker and the next concrete fix.

## Source Anchors

These local rules follow the same general patterns as:

- OpenAI's practical agent guidance on keeping agent orchestration, guardrails,
  and human oversight explicit:
  https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
- Anthropic's Claude Code memory guidance on keeping project instructions in
  persistent repository memory:
  https://docs.anthropic.com/en/docs/claude-code/memory
- ReAct's observe/reason/act pattern for tool-using agents:
  https://arxiv.org/abs/2210.03629
