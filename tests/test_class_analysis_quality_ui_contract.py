from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "ybat-master" / "tator.html").read_text(encoding="utf-8")
JS = (ROOT / "ybat-master" / "ybat.js").read_text(encoding="utf-8")
CSS = (ROOT / "ybat-master" / "class_split_controls.css").read_text(encoding="utf-8")


def _extract_js_function(source, name):
    start = source.index(f"function {name}")
    parameter_start = source.index("(", start)
    parameter_depth = 0
    parameter_end = None
    for index in range(parameter_start, len(source)):
        if source[index] == "(":
            parameter_depth += 1
        elif source[index] == ")":
            parameter_depth -= 1
            if parameter_depth == 0:
                parameter_end = index
                break
    if parameter_end is None:
        raise AssertionError(f"Could not extract {name} parameters")
    brace_start = source.index("{", parameter_end)
    depth = 0
    for index in range(brace_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"Could not extract {name}")


class ClassAnalysisQualityUiContractTests(unittest.TestCase):
    def test_quality_recipe_and_deep_evidence_controls_are_explicit(self):
        self.assertIn('value="thorough_quality_v1" selected', HTML)
        self.assertIn('value="precise_compact_v1"', HTML)
        self.assertIn('value="fast_map_v1"', HTML)
        self.assertIn("Spatial evidence refinement", HTML)
        self.assertIn("does not ask a VLM to judge labels", HTML)
        self.assertIn('id="classSplitRefineOutliers"', HTML)
        self.assertIn("classSplitRefinementDefaultForPreset", JS)

    def test_tiny_and_adaptive_controls_are_present(self):
        self.assertIn('id="classSplitSizeFilter"', HTML)
        self.assertIn('value="low_detail_only"', HTML)
        self.assertIn('value="low_detail">Tiny / low-detail boxes', HTML)
        self.assertIn('value="relative_small"', HTML)
        self.assertIn('id="classSplitLowDetailMinSide"', HTML)
        self.assertIn('id="classSplitAdaptiveRanking"', HTML)
        self.assertIn('id="classSplitAdaptiveRankingUpdate"', HTML)
        self.assertIn('id="classSplitAdaptiveRankingReset"', HTML)

    def test_memory_budget_queue_and_adaptive_race_contracts_are_explicit(self):
        self.assertIn('id="classSplitQualityMemoryPolicy"', HTML)
        self.assertIn('value="auto" selected', HTML)
        self.assertIn('id="classSplitQualityMemoryBudget"', HTML)
        self.assertIn('id="classSplitQualityReviewFraction"', HTML)
        self.assertIn("quality_memory_policy", JS)
        self.assertIn("quality_memory_budget_mb", JS)
        self.assertIn("quality_review_fraction", JS)
        self.assertIn("adaptiveRankingOperation", JS)
        self.assertIn("adaptiveRankingOrderById", JS)
        self.assertIn("classSplitAdaptiveRankingOperationIsCurrent", JS)
        self.assertIn("preflightClassSplitRequest", JS)
        self.assertIn("quality_full_warning_acknowledged", JS)
        self.assertIn("quality_review_queue", JS)
        self.assertIn("Show all flagged objects", HTML)

    def test_proposed_class_is_explicit_and_never_automatic(self):
        self.assertIn("proposed_class", JS)
        self.assertIn("Use proposed class", JS)
        self.assertIn("apply-proposed-class", JS)
        self.assertNotIn("autoApplyProposedClass", JS)

    def test_startup_operation_can_build_its_own_request_only(self):
        mutation_busy = _extract_js_function(JS, "classSplitMutationIsBusy")
        recovery_reason = _extract_js_function(
            JS, "classSplitRecoveryMutationBlockReason"
        )
        write_blocked = _extract_js_function(
            JS, "classSplitWriteMutationIsBlocked"
        )
        build_request = _extract_js_function(JS, "buildClassSplitRequest")
        start_analysis = _extract_js_function(JS, "startClassSplitAnalysis")
        self.assertIn("ignoreStartupOperationToken: startupOperationToken", build_request)
        self.assertIn(
            "buildClassSplitRequest({ startupOperationToken: startupToken })",
            start_analysis,
        )
        script = "\n".join([
            "const assert = require('assert');",
            "const classSplitState = {startupOperation: {token: 17}, multiSelectionActionInFlight: false, adaptiveRankingOperation: null, relabelInFlight: false, reviewActionPendingPointIds: new Set(), reviewCommitDrainPromise: null, reviewDispositionReconciliationPointIds: new Set(), reviewDispositionHydrationTimer: null, reviewDispositionHydrationTargets: new Map(), reviewDispositionHydrationInFlight: new Set(), currentJobId: ''};",
            "const annotationSourceState = {saveInFlight: false, dirtyRecordsByKey: new Map()};",
            "function classSplitReviewDispositionInFlightForJob() { return false; }",
            "function classSplitReviewHistoryDeleteOperation() { return null; }",
            "function classSplitPendingReviewCommitCountForJob() { return 0; }",
            mutation_busy,
            recovery_reason,
            write_blocked,
            "assert.strictEqual(classSplitMutationIsBusy({ignoreStartupOperationToken: 17}), false);",
            "assert.strictEqual(classSplitMutationIsBusy({ignoreStartupOperationToken: 18}), true);",
            "assert.strictEqual(classSplitMutationIsBusy(), true);",
            "classSplitState.startupOperation = null;",
            "assert.strictEqual(classSplitMutationIsBusy(), false);",
            "classSplitState.currentJobId = 'job-1';",
            "classSplitState.annotationRecovery = {job_id: 'job-1', status: 'rerun_required', rerun_required: true};",
            "assert.match(classSplitRecoveryMutationBlockReason(), /read-only/);",
            "assert.strictEqual(classSplitWriteMutationIsBlocked(), true);",
            "assert.strictEqual(classSplitMutationIsBusy(), false);",
        ])
        subprocess.run(["node", "-e", script], cwd=ROOT, check=True)

    def test_review_commit_state_distinguishes_unsent_rejected_and_unknown(self):
        commit_state = _extract_js_function(
            JS, "classSplitReviewDispositionCommitState"
        )
        script = "\n".join([
            "const assert = require('assert');",
            commit_state,
            "assert.strictEqual(classSplitReviewDispositionCommitState(new Error('preflight')), 'not_sent');",
            "assert.strictEqual(classSplitReviewDispositionCommitState({reviewDispositionCommitState: 'rejected'}), 'rejected');",
            "assert.strictEqual(classSplitReviewDispositionCommitState({reviewDispositionCommitUnknown: true}), 'unknown');",
            "assert.strictEqual(classSplitReviewDispositionCommitState({}, 'unknown'), 'unknown');",
        ])
        subprocess.run(["node", "-e", script], cwd=ROOT, check=True)

    def test_terminal_recovery_blocks_pair_vlm_and_adaptive_mutations(self):
        envelope = _extract_js_function(
            JS, "classSplitTerminalRecoveryEnvelope"
        )
        transition = _extract_js_function(
            JS, "transitionClassSplitToRerunRequired"
        )
        merge_recovery = _extract_js_function(
            JS, "mergeClassSplitAnnotationRecovery"
        )
        adaptive_update = _extract_js_function(
            JS, "updateClassSplitAdaptiveRanking"
        )
        adaptive_reset = _extract_js_function(
            JS, "resetClassSplitAdaptiveRanking"
        )
        qwen_start = _extract_js_function(
            JS, "startClassSplitQwenReview"
        )
        pair_commit = _extract_js_function(
            JS, "commitClassSplitDualBBoxDeletionTransaction"
        )
        controls = _extract_js_function(
            JS, "refreshClassSplitControls"
        )
        self.assertIn("classSplitWriteMutationIsBlocked", adaptive_update)
        self.assertIn("classSplitWriteMutationIsBlocked", adaptive_reset)
        self.assertIn("classSplitRecoveryMutationBlockReason", qwen_start)
        self.assertIn("expected_entity_record_revision", pair_commit)
        self.assertIn("entity_preconditions", pair_commit)
        self.assertIn(
            "transitionClassSplitToRerunRequired",
            pair_commit,
        )
        self.assertIn("writeMutationBlocked", controls)
        self.assertIn(
            '"class-analysis-dual-bbox-annotation-commit-v2"',
            JS,
        )
        self.assertIn(
            "dual_bbox_annotation_transaction_api_version\n"
            "            ) >= 5",
            JS,
        )
        self.assertIn("dual_bbox_delete_v1", pair_commit)
        self.assertIn("queueClassSplitAnnotationTransaction", pair_commit)
        self.assertIn("reconcileClassSplitAnnotationTransactionAfterFailure", pair_commit)
        drain = _extract_js_function(
            JS, "drainClassSplitPendingAnnotationTransactions"
        )
        self.assertIn("dispatchRequest", drain)
        self.assertIn("reconcileClassSplitAnnotationTransactionAfterFailure", drain)
        self.assertIn("classSplitSemanticAnnotationTransactionRequest", JS)
        script = "\n".join([
            "const assert = require('assert');",
            "const classSplitState = {currentJobId: 'job-new', analysisGeneration: 9, annotationRecovery: null};",
            "function parseJsonObjectSafe(value, fallback) { try { return JSON.parse(value); } catch (_) { return fallback; } }",
            "function classSplitAsyncRequestIsCurrent(generation, jobId) { return generation === classSplitState.analysisGeneration && jobId === classSplitState.currentJobId; }",
            "function renderClassSplitSessionPersistenceStatus() {}",
            "function refreshClassSplitControls() {}",
            "function renderClassSplitWrongList() {}",
            "function renderClassSplitReport() {}",
            envelope,
            merge_recovery,
            transition,
            "assert.strictEqual(transitionClassSplitToRerunRequired({detail: {status: 'rerun_required', rerun_required: true, job_id: 'job-old'}}, {jobId: 'job-old', generation: 8}), false);",
            "assert.strictEqual(classSplitState.annotationRecovery, null);",
            "assert.strictEqual(transitionClassSplitToRerunRequired({detail: {code: 'annotation_entity_changed_rerun_required', job_id: 'job-new', reason: 'stale'}}, {jobId: 'job-new', generation: 9}), true);",
            "assert.strictEqual(classSplitState.annotationRecovery.status, 'rerun_required');",
            "assert.strictEqual(classSplitState.annotationRecovery.reason, 'stale');",
            "const retained = mergeClassSplitAnnotationRecovery(classSplitState.annotationRecovery, {status: 'ready', checkpoint_ready: true}, 'job-new');",
            "assert.strictEqual(retained.status, 'rerun_required');",
            "assert.strictEqual(retained.rerun_required, true);",
        ])
        subprocess.run(["node", "-e", script], cwd=ROOT, check=True)

    def test_default_setup_is_compact_and_advanced_evidence_is_disclosure_only(self):
        self.assertIn('id="classSplitGuidedSetup"', HTML)
        self.assertIn('id="classSplitStepScopeTitle"', HTML)
        self.assertIn('id="classSplitStepMapTitle"', HTML)
        self.assertIn('id="classSplitStepFeaturesTitle"', HTML)
        self.assertIn('id="classSplitStepMemoryTitle"', HTML)
        self.assertIn('id="classSplitStepRefineTitle"', HTML)
        self.assertIn('id="classSplitStepRunTitle"', HTML)
        self.assertIn('id="classSplitClassField"', HTML)
        self.assertIn('id="classSplitClassSelect"', HTML)
        self.assertIn('<details class="class-split-tune" id="classSplitFeatureTuning"', HTML)
        self.assertNotIn('id="classSplitFeatureTuning" open', HTML)
        self.assertNotIn('id="classSplitRefinementPreview" open', HTML)
        self.assertIn(".class-split-guided-setup", CSS)
        self.assertIn("html.theme-dark .class-split-workspace", CSS)
        self.assertIn("html.theme-pipboy .class-split-workspace", CSS)
        self.assertIn("body:has(#tabClassSplit.active) .ui-tooltip", CSS)
        self.assertIn(".class-split-embedding-guide > summary::before", CSS)


if __name__ == "__main__":
    unittest.main()
