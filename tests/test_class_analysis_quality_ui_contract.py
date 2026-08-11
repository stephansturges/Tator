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
        self.assertIn("Deep evidence", HTML)
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
            "assert.strictEqual(classSplitMutationIsBusy({ignoreStartupOperationToken: 17}), false);",
            "assert.strictEqual(classSplitMutationIsBusy({ignoreStartupOperationToken: 18}), true);",
            "assert.strictEqual(classSplitMutationIsBusy(), true);",
            "classSplitState.startupOperation = null;",
            "assert.strictEqual(classSplitMutationIsBusy(), false);",
        ])
        subprocess.run(["node", "-e", script], cwd=ROOT, check=True)

    def test_default_setup_is_compact_and_advanced_evidence_is_disclosure_only(self):
        self.assertIn('id="classSplitAdvancedSetup"', HTML)
        self.assertIn('class="class-split-grid class-split-grid--primary"', HTML)
        primary = HTML[
            HTML.index('class="class-split-grid class-split-grid--primary"'):
            HTML.index('id="classSplitAdvancedSetup"')
        ]
        self.assertIn('id="classSplitClassField" hidden', primary)
        self.assertIn('id="classSplitClassSelect"', primary)
        self.assertIn('<details id="classSplitEmbeddingGuide"', HTML)
        self.assertNotIn('class="class-split-embedding-guide__preview" open', HTML)
        self.assertNotIn('class-split-field--wide embedding-recipe-note" open', HTML)
        self.assertIn('class="class-split-run-strip"', HTML)
        self.assertIn("max-height: min(44vh, 360px)", CSS)
        self.assertIn("html.theme-dark .class-split-workspace", CSS)
        self.assertIn("html.theme-pipboy .class-split-workspace", CSS)
        self.assertIn("body:has(#tabClassSplit.active) .ui-tooltip", CSS)
        self.assertIn(".class-split-embedding-guide > summary::before", CSS)


if __name__ == "__main__":
    unittest.main()
