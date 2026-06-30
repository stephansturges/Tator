# Qwen Labeling Research Roadmap

Status: local research note, intentionally not committed.

## North Star

Use Qwen as the annotation copilot: generate labels, review likely annotation
errors, explain uncertainty, and improve over time by training on accepted
human-corrected datasets.

The target loop is:

1. Specialized spatial engines generate candidate boxes, points, or masks.
2. Qwen reasons over source-image context, class glossary, examples, overlaps,
   grounding proposals, and user guidance.
3. The human accepts, fixes, skips, or confirms.
4. Accepted annotations and decisions become training data.
5. Qwen adapters are trained and benchmarked against repeatable labeling tasks.

## Priority Stack

1. Canonical Qwen dataset export and evaluation loop.
   - Stable train/validation/test exports from Tator annotations.
   - Metrics for valid JSON, class accuracy, box IoU, missed objects,
     hallucinated objects, useful proposal rate, and review action rate.
   - This is the scoreboard for all model/runtime changes.

2. Qwen3.6 inference hardening.
   - Wire official, abliterated, and quantized Qwen3.6 candidates into every
     Qwen usage surface that can support them.
   - Keep local VLM final reasoning central in likely-wrong review.
   - Treat runtime failures as runtime/model issues, not as reasons to replace
     Qwen with deterministic controller fallback.

3. LocateAnything / PBD inference as a grounding engine.
   - PBD belongs beside Qwen, not inside the first Qwen fine-tuning milestone.
   - Use it to propose boxes/points and to answer grounding queries like
     "locate all objects matching class X."
   - Feed PBD results to Qwen as structured visual evidence.

4. Qwen-agent plus PBD tools.
   - Let Qwen request spatial evidence for current class, suggested class, and
     ambiguous overlap cases.
   - Use grounding proposals to reduce blind visual reasoning and reduce skips.
   - Human-controlled mutation remains the default.

5. Qwen adapter training.
   - Train LoRA/QLoRA adapters on accepted human annotations and corrected
     likely-wrong decisions.
   - CUDA QLoRA is the serious training path for large models.
   - MLX adapter training is useful for Apple Silicon but must be smoke-tested
     per architecture and model family.

6. Abliterated and quantized model support.
   - Abliterated checkpoints can be useful for local review behavior.
   - Quantized inference models should map cleanly to trainable base models for
     QLoRA, or be produced after training.
   - Avoid pretending AWQ/GPTQ/FP8 inference artifacts are the normal target for
     full-weight training.

7. SwiReasoning as an optional inference/decoding mode.
   - SwiReasoning is a training-free switch between explicit and latent
     reasoning.
   - It should be benchmarked as a decoding mode for hard review cases.
   - It may improve reasoning depth, but may also degrade strict JSON behavior.

8. PBD fine-tuning from Tator datasets.
   - Worth doing after PBD inference proves useful.
   - This is CUDA-first and heavier than Qwen adapter training.

9. PBD inside Qwen3.6 architecture.
   - Research-project scale.
   - Not a near-term product implementation target.

## Recommended Architecture

For labeling:

```text
Image
  -> LocateAnything/PBD proposals
  -> optional YOLO/RF-DETR/SAM/SAM3 proposals
  -> Qwen review and consolidation
  -> human accept/fix
  -> Tator annotation state
  -> training export
  -> Qwen adapter fine-tune
```

For likely-wrong review:

```text
Likely-wrong object
  -> source crop + context crop + whole-image/context views
  -> same-class examples
  -> suggested-class examples
  -> overlap, scale, embedding, and quality diagnostics
  -> optional PBD grounding query for current/suggested classes
  -> Qwen final judgment
  -> human-controlled apply/confirm/skip
```

## Close-Term Focus: Qwen3.6 + SwiReasoning

The most promising near-term experiment is:

1. Harden Qwen3.6 inference for normal generation.
2. Add SwiReasoning as an optional CUDA/Transformers decoding backend where
   possible.
3. Benchmark normal Qwen3.6 versus SwiReasoning Qwen3.6 on the same likely-wrong
   vignette and labeling tasks.
4. Keep raw prompts, image packs, generated outputs, parsed JSON, and human audit
   notes in benchmark artifacts.

### What Is Missing Before Testing

1. Transformers/runtime support for official Qwen3.6.
   - Official Qwen3.6 uses the `qwen3_5_moe` architecture.
   - The local pinned Transformers path currently does not recognize that model
     type in ordinary `AutoConfig` loading.
   - We need either a safe Transformers upgrade/source install, an isolated
     Qwen3.6 runtime environment, or an OpenAI-compatible external server path
     such as `transformers serve`/vLLM where available.

2. A Qwen3.6 model-selection policy.
   - Standard official Qwen3.6 for baseline.
   - Abliterated Qwen3.6 for review behavior.
   - Quantized Qwen3.6 for local feasibility.
   - Each entry needs metadata for runtime, precision, vision support, known
     failures, and training eligibility.

3. A normal-generation Qwen3.6 smoke harness.
   - Single image caption.
   - Single strict-JSON detection/review prompt.
   - Class-split likely-wrong finalization prompt.
   - Same tests for official, abliterated, and quantized variants.

4. SwiReasoning adapter/wrapper.
   - The reference implementation targets text causal LMs and manually drives
     generation using embeddings, entropy, and KV cache.
   - We need a wrapper that can run after Qwen-VL has encoded the prompt/images,
     or a first-stage text-only harness to prove it on our final reasoning step.
   - MLX-VLM support is not immediate because SwiReasoning depends on internals
     that are easy to access in PyTorch/Transformers but not currently exposed
     through our MLX generation wrapper.

5. Strict-output compatibility layer.
   - SwiReasoning can create longer or more free-form reasoning traces.
   - Our product path needs compact parseable JSON.
   - The test harness must measure valid JSON rate, schema adherence, and useful
     action rate, not only subjective reasoning quality.

6. Benchmark switchboard.
   - Same examples, same image packs, same class glossary, same model prompt.
   - Compare:
     - Qwen3.6 normal decoding.
     - Qwen3.6 with longer token budget.
     - Qwen3.6 with explicit CoT/hidden thinking prompt policy.
     - Qwen3.6 with SwiReasoning.
     - Current Qwen3.6 MLX path where available.

7. Artifact retention.
   - Save prompt, tool evidence, image bundle manifest, raw model output, parsed
     result, controller decision, and human audit result.
   - This prevents drift back into "safe but useless" controller-only behavior.

## References

- SwiReasoning: https://github.com/sdc17/SwiReasoning
- SwiReasoning paper: https://arxiv.org/abs/2510.05069
- Qwen3.6 35B-A3B: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.6 abliterated MLX candidate:
  https://huggingface.co/vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit
- LocateAnything / PBD: https://github.com/NVlabs/Eagle/tree/main/Embodied
- LocateAnything paper: https://arxiv.org/abs/2605.27365
