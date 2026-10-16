Dataset pruning
===============

Dataset pruning proposes a smaller dataset, shows what would be lost, and
exports the images remaining after approved removals as a new dataset variant.
It does not delete source images or edit their annotations.

The planning and review rules below apply to both applications. The getting
started steps and unprefixed API routes describe the local workbench. Hosted
administrators start a pruning task from **Quality & compute**, then open its
completed result for review; inputs are frozen confirmed annotations and their
captured working media. See
`Hosted analysis and training <hosted_analysis.rst>`_ for that entry point and
the `documentation index <index.rst>`_ for other workflows.

Getting started
---------------

1. Open a dataset and save its annotations. Open **Dataset pruning**, beside
   the Data Quality Explorer (DQE).
2. Choose how to describe whole images and, for compound analysis, labelled
   objects. You can inherit the DQE recipe or select compatible components.
   The component table shows the actual models, weights and size corrections.
3. Analyze the saved source. Compatible cached descriptors are reused. Missing
   components may require encoding; the interface reports that preparation.
4. Choose **Strict duplicates** or **Allow a coverage budget**. Adjust the
   controls, inspect the preview, then apply it. Applying a plan does not
   approve any removals.
5. Inspect the proposed images. Keep valuable views, approve removals individually
   or through the family controls, and export the approved subset.

Changing the planning controls reuses the completed analysis. It does not run
the image encoders, retrain SALAD or refit size correction. Changing the feature
recipe, labels or matching thresholds is a different operation: the relevant
derived analysis must be rebuilt or loaded from a compatible cache.

The four planning controls
--------------------------

**Object importance** changes the balance between whole-image similarity and
object appearance when ranking possible removals in the coverage-budget mode.
A larger value gives more influence to object variety. It does not lower the
individual matching thresholds. In strict mode, it also changes the compound
similarity score used to qualify a pair for removal.

**Maximum images removed** sets a ceiling, displayed as both an exact count
and a percentage of the dataset. The planner may propose fewer images when
the other constraints prevent further removals.

**Maximum affected appearance groups** limits how many distinct appearance
groups may lose coverage. This is a count, not a count of physical objects.

**Maximum rarity-weighted loss** limits the affected groups' share of the
dataset's total appearance weight. Rarer classes and unusual appearances
within a class receive more weight. Both loss limits apply at the same time;
both start at zero.

The coverage-budget policy requires compound image/object features and native
scale protection. Strict duplicate pruning remains available for existing
workflows. In coverage-budget mode, a permitted loss budget can remove every
example of a class; the interface reports this consequence. This mode has no hidden class-count floor.
Images you explicitly protect and confirmed family keepers must remain.

What counts as a loss?
----------------------

In coverage-budget mode, the planner checks every original labelled object
against the images that would remain. A replacement must have the same class,
sufficiently similar appearance, and compatible native detail under the
configured scale rule.
Different objects in one removed image may have replacements in different
retained images. Every removed image also needs a directly matching retained
whole image. An explicitly negative image needs a negative-image replacement.
Strict compound pruning instead requires one retained image to cover the
removed image's objects under its pair-matching rules.

Appearance groups are fixed counting units built from object embeddings.
They use fixed exemplars rather than merging chains of similar objects.
If even one member has no qualifying retained replacement, its group counts
as affected, once. Removing ten repeated annotations can therefore affect no
groups, one group, or several groups, depending on what remains elsewhere.

These groups estimate visual variety under the chosen recipe. They do not
establish that two annotations depict the same physical vehicle or person.
The interface distinguishes removed annotation occurrences, uncovered objects,
and affected appearance groups. A zero-loss plan means zero loss under these
matching rules, not proof that no useful training information was removed.

Rarity weights and group membership are frozen before planning. Moving a
slider cannot redefine the denominator to make a loss appear smaller. For a
group in class ``c``, its weight is::

    class_weight = clip(sqrt(median_class_group_count / class_group_count[c]), 0.5, 2)
    group_weight = class_weight * (1 + within_class_isolation_percentile)
    loss_fraction = sum(weight of affected groups) / sum(weight of all groups)

Isolation uses mean embedding distance to up to five nearest other exemplars
of the same class, converted to a within-class midrank percentile. A singleton
class uses percentile one. Class frequency is measured in appearance groups,
so repeated frames do not each add a separate rarity unit.

Reviewing and changing a plan
-----------------------------

The graph and counts describe the current ready preview. **Apply** captures
that exact proposal. **Discard** returns to the applied settings. Superseded
or unfinished previews cannot approve images or drive an export.

The impact view brings costly losses forward and shows the actual object
crops. **Keep** protects an image and shrinks the current removal set. It does
not silently fill the vacancy with another image. Removing that protection
does not restore the old proposal either: use **Recalculate** to request a
new plan. Exact-image Keep decisions carry across analyses of the same source.

Families collect related views for review. The bulk control means **keep at
least X images per family**; the matching and loss rules can require more.
Review the resulting preview before accepting its captured list of removals.
A chain of similarities does not justify removing an image: its replacement
must be directly qualified and remain in the final set.

Approvals bind to a particular plan, source and evidence generation. A changed
plan may invalidate earlier approvals. Undo restores only changes still owned
by that operation, preserving intervening Keep decisions. If a request loses
its response, reconnect to recover the saved operation's receipt; the browser
does not guess whether it committed.

Exporting a variant
-------------------

Export removes only approved images. The loss report is recomputed for that
actual subset, which can differ from the complete proposal. Local exports include
retained source image bytes and saved labels/captions. Hosted exports use the
analysis's frozen working-image bytes and confirmed annotation versions; those
working images can differ from original uploads when pixel conversion was
required. Original uploaded bytes and provenance remain in hosted storage.
The export checks the source and review generation again before publishing
the download.

The grouped split option keeps duplicate/video groups together when rebuilding
train/validation splits. The preserve option keeps existing split membership.
Neither option alters the source dataset.

Reusing DQE and SALAD
---------------------

Pruning shares DQE's object-feature producer and ingestion's whole-image
descriptor machinery. Recipes can combine multiple backbones, their existing
size-and-shape adjustments, and locally trained SALAD heads. The selected
recipe is recorded with the job; this feature does not replace saved recipes
with a new universal default.

SALAD profiles retain dataset association, representation scope, encoder,
training provenance and head identity. Whole-image and crop heads are not
interchangeable. Renaming or archiving a profile changes its management state;
it does not silently retrain or replace its weights.

Descriptor cache keys bind source pixels, crop geometry where applicable,
encoder/preprocessing identity, SALAD weights and numerical implementation.
Fitted size corrections are reused only with their recorded identity. Native
size/occupancy checks remain separate from correcting embedding size bias.

The optional C-RADIO SALAD RMS input policy scales each patch before the learned
aggregation branches. Its ``local-salad-v2-rms-inputs`` head format prevents an
older reader from silently ignoring that policy. Existing raw-input heads
retain their original behavior. This option is not a claim that one recipe
works best on every dataset or runtime.

Implementation and API
----------------------

``api/dataset_pruning.py`` defines the ``/dataset_pruning`` routes.
``dataset_pruning_adapter.py``, ``producer.py``, ``components.py`` and
``cache.py`` connect the shared feature machinery to frozen source snapshots.
``coverage.py`` owns direct replacement checks, appearance groups and loss
accounting; ``planning.py`` and ``jobs.py`` coordinate previews and job state.
``review.py`` stores the authoritative review ledger, operation receipts and
plan revisions. ``evidence.py`` and ``serving.py`` bind displayed evidence to
the source. These modules live under ``services/`` with the
``dataset_pruning_`` prefix.

The shared storage coordinator protects source reads, saved reviews, SALAD
heads and in-progress exports from concurrent cleanup. Automatic cleanup is
limited to unused, regenerable caches; saved work requires explicit removal.
The browser binds pruning requests to the backend's persistent identity and
rejects a replacement server at the same URL until the workspace is reopened.

The main sequence is::

    POST /dataset_pruning/jobs
    GET  /dataset_pruning/jobs/{job_id}/result
    POST /dataset_pruning/jobs/{job_id}/plan/preview
    GET  /dataset_pruning/jobs/{job_id}/plan/previews/{preview_id}
    POST /dataset_pruning/jobs/{job_id}/review
    POST /dataset_pruning/jobs/{job_id}/exports

Hosted review mounts these routes beneath
``/api/hosted/analysis/jobs/{hosted_job_id}``. Creating a new hosted analysis
uses the configured ``/api/hosted/tools/heavy`` task; the mounted pruning
creation route cannot start a new producer run. Hosted session and snapshot
checks apply in addition to the shared review ledger.

Settings previews use ``kind=settings`` and the five fields in ``PlanSettings``:
``policy``, ``object_weight``, ``max_removed_images``,
``max_affected_appearance_groups`` and
``max_lost_appearance_weight_fraction``. Mutations require the captured
selection/review/plan revisions and a stable request ID. Apply uses the review
action ``apply_plan`` with the ready preview ID. Use the typed route models
and generated API schema for complete request fields.

Nearest-neighbor shortlists guide ranking; their absence is not proof of
novelty. Accepted replacement witnesses and losses use authoritative exact
checks against the frozen source. Sparse indexes, memory-mapped features and
disk-backed certificates avoid a dense all-pairs matrix. Live indexes are
owned by their job directory, even when source and recipe hashes match.

Coverage schema ``pruning-direct-appearance-coverage-v3`` freezes the accounting
definition. Qualification ``authoritative-direct-witness-v2`` distinguishes
the repaired exact-witness checks. Historical coverage-budget previews,
approvals and prepared exports that lack the current qualification need a
fresh preview, Apply and approval; they are not silently promoted to the
new generation.

Verification and limits
-----------------------

The focused regression suite covers direct witnesses, genuine permitted loss,
negative-only sources, cache ownership, revision races, idempotent retries,
Undo, inherited Keep decisions and approved-subset export::

    python -m pytest -q tests/test_dataset_pruning_coverage.py \
      tests/test_dataset_pruning_planning.py tests/test_dataset_pruning_cache_replay.py

The local pruning repair review recorded 102 passing tests. A separate isolated
browser run used 47 copied WALDO40 images and exercised preview,
lost-response recovery, Keep,
approval, reload and a 46-image export. Eighteen cached full-dataset plans were
checked on 11,572 images and 259,461 objects without modifying the originals.
These local-workbench checks establish the exercised behavior, not an accuracy
estimate, a detector-training benefit or hosted GPU qualification. Hosted
verification boundaries are described in
`Hosted analysis and training <hosted_analysis.rst>`_.

The planner is greedy and does not promise a globally optimal retained set.
Initial accounting can be expensive. Warm planning on the measured WALDO40
snapshot took about 0.30–0.39 seconds, while complete warm API previews took
about 2.25–2.27 seconds after the first request: the two-second API target was
not met. The live coverage-index map is not yet bounded, and prolonged
multi-process contention has not been stress-tested. Loading this code in a
running deployment requires a backend restart and a fresh browser page.
