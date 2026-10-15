Baseline before the interface redesign
======================================

This inventory records the source consolidation on 15 October 2026. It
distinguishes delivered functionality from experiments and deployment tasks,
so the redesign can preserve useful work without inheriting stale plans.

Delivered source
----------------

* Dataset pruning, including cached multi-backbone/SALAD reuse, strict
  duplicates, explicit appearance-loss budgets, post-analysis object weights,
  review, family keepers and approved-subset export. Its workflow and measured
  limits are in `Dataset pruning <dataset_pruning.rst>`_.
* The shared storage catalog, dependency-aware retention, saved review
  recovery and the DQE **Datasets & sessions** interface. The final integration
  retains pruning's backend-identity checks. See
  `Datasets and saved review sessions <dataset_library.rst>`_.
* The existing macOS startup cleanup and classifier compatibility pin. See
  `macOS runtime defaults <macos_runtime_defaults.rst>`_.

No product functionality is removed in this consolidation. Backend activation
is a separate operational step: updating Git does not restart a running
process. An updated backend and a browser reload are required to use the new
library interface with the current pruning code.

Known limits and further work
----------------------------

**Pruning performance.** The latest full API measurement was about 2.25–2.27
seconds after warm-up, above the two-second target. The live coverage-index
map is not bounded, and sustained write contention has not been stress-tested.
Selection is a greedy heuristic. Neither matching-rule coverage nor the
selected visual examples establish improved detector training performance.

**Detector-assisted ingestion.** The ten-image incoming-data experiment in
the explanatory report is complete. Turning that experiment into a detector
selection and compound-diversity workflow in the ingestion tool remains a
product task. It is not an unfinished branch or a delivered production flow.

**Background evidence.** An auxiliary background-likeness signal for DQE was
proposed but not experimentally evaluated. Existing background classifiers
and spatial refinement remain as implemented. A separate comparison is
needed before changing the quality recipe.

**C-RADIO/SALAD precision.** Per-patch RMS input scaling and its versioned
head format are implemented. The report's strict default-MLX parity check
failed; the full-precision reference passed. Small measured pair-score
differences do not establish universal numerical equivalence or pruning
accuracy. Existing heads retain their original preprocessing policy.

**Report publication.** The explanatory report and separate WALDO40 gallery
are packaged into three static-hosting ZIPs below 35 MB each. Public hosting
upload has not been performed. The report builder, accepted assets, private
data and evidence remain local artifacts, outside the product repository.

Historical plans for guided setup, recovery transport and C-RADIO acceleration
contain completed delivery records despite earlier unchecked lists. They do
not represent stranded source branches. Optional descriptor comparisons and
larger benchmark sweeps remain research ideas, not delivered capabilities.

Repository boundaries
---------------------

Product code, existing tests and user documentation belong in Git. Original
datasets, annotations, model weights, generated reports, ZIPs, credentials,
local agent state and historical working notes remain excluded. A clean Git
working tree does not mean those local assets have been deleted or backed up
to the source remote.

The consolidation audit found one local worktree on ``main``, no stashes,
only ``main`` on the remote, and no open pull requests. No branch deletion
or unrelated merge was needed. Functional cleanup should start as a new
change against this consolidated baseline.
