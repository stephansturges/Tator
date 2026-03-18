# Ensemble Detection Recipe (EDR) Explainer

## What an EDR is

An **Ensemble Detection Recipe (EDR)** is the full reusable detection workflow in Tator.

It has two parts:

- the **prepass**, which generates a broad candidate pool
- the **calibration**, which scores and filters that candidate pool

The prepass is responsible for recall. The calibration is responsible for precision and final acceptance.

## Prepass structure

The prepass is a deterministic pipeline that combines several sources of evidence:

- **Detectors**:
  YOLO and RF-DETR provide the main seed boxes. They are usually the strongest source for geometry and for common classes.

- **Windowed detector passes**:
  These rerun detection on tiles instead of only the full image. They help recover small objects and boundary cases that are easy to miss at full-frame scale.

- **SAM3 text**:
  SAM3 text uses glossary terms to expand recall beyond what the detectors found. This is valuable for classes the detectors under-call, but it is usually noisier than detector output.

- **Dedupe and fusion**:
  Multiple sources often point at the same object. Dedupe merges those overlaps into one candidate while keeping the evidence that created it, including source list, scores, and atom provenance.

- **SAM3 similarity**:
  Similarity starts from confident exemplars and looks for visually similar objects. It is especially useful when objects repeat across the image. In practice it is often a safer recall extender than broad text expansion.

The output of the prepass is not the final answer. It is a candidate set with rich provenance.

## What each component contributes

| Component | Main contribution | Main tradeoff |
|---|---|---|
| Detectors | Strong seed boxes, class anchors | Misses small or difficult objects |
| Windowed passes | Better small-object and edge recall | Slower, more duplicates |
| SAM3 text | Recovers detector misses using label language | More false positives |
| Dedupe/fusion | Turns many atoms into one candidate with provenance | Sensitive to merge settings |
| SAM3 similarity | Recovers repeated look-alikes from exemplars | Can drift if exemplars are weak |

## How the calibrator works

The calibrator does not invent objects or move boxes around. It only decides which existing candidates should survive.

For each candidate, the system builds a feature row from:

- source provenance
- detector and SAM scores
- source agreement and overlap signals
- geometry and cluster structure
- optional classifier probabilities and embeddings

Candidates are then matched to ground truth on a sampled labeled image set. That produces training labels for the calibrator.

The current default calibrator is XGBoost. Its role is to learn a score that separates likely true objects from likely false positives.

## How tuning works

Training the model is only part of the process. After training, the pipeline still needs to decide how strict to be.

Threshold tuning searches per-class acceptance cutoffs on the validation split. It tries to satisfy the requested recall or false-positive constraints while optimizing the selected metric, usually F1.

EDR discovery may also compare larger structural choices, including:

- lane choice, such as windowed vs non-windowed
- scenario settings, such as optional quality heads
- policy settings, such as source-specific biases or consensus requirements

A change is only promoted if it clears the defined comparison gate against the current baseline. Small or unstable wins are rejected.

## What the final EDR contains

A promoted EDR stores:

- the chosen lane
- the chosen scenario settings
- the chosen policy settings
- the calibrator configuration and expected metrics

That lets later jobs reuse the same proven setup instead of rediscovering it every time.
