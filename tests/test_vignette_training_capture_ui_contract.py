import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_HTML_PATH = REPO_ROOT / "ybat-master" / "tator.html"
DESKTOP_JS_PATH = REPO_ROOT / "ybat-master" / "ybat.js"
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_js_function(source: str, name: str) -> str:
    start = source.index(f"function {name}")
    params_start = source.index("(", start)
    params_depth = 0
    params_end = -1
    quote = ""
    escaped = False
    for index in range(params_start, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "(":
            params_depth += 1
        elif char == ")":
            params_depth -= 1
            if params_depth == 0:
                params_end = index
                break
    assert params_end >= 0, f"Could not parse JavaScript parameters for {name}"
    brace_start = source.index("{", params_end)
    depth = 0
    quote = ""
    escaped = False
    for index in range(brace_start, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Could not extract JavaScript function {name}")


def _input_tag(html: str, element_id: str) -> str:
    match = re.search(
        rf"<input\b[^>]*\bid=[\"']{re.escape(element_id)}[\"'][^>]*>",
        html,
        flags=re.IGNORECASE,
    )
    assert match is not None, f"missing input #{element_id}"
    return match.group(0)


def _assert_default_unchecked(input_tag: str) -> None:
    assert re.search(r"\btype=[\"']checkbox[\"']", input_tag, flags=re.IGNORECASE)
    assert re.search(r"\bchecked(?:\s|=|/|>)", input_tag, flags=re.IGNORECASE) is None


def test_desktop_training_capture_is_an_explicit_page_local_opt_in():
    html = _read(DESKTOP_HTML_PATH)
    js = _read(DESKTOP_JS_PATH)

    capture_input = _input_tag(html, "classSplitTrainingCapture")
    _assert_default_unchecked(capture_input)
    assert 'autocomplete="off"' in capture_input
    assert "Save decisions and linked evidence for future training" in html
    assert "Opt in for this page only." in html
    assert "Separate append-only local training-candidate corpus." in html

    reset_capture = _extract_js_function(js, "resetClassSplitTrainingCapture")
    init_explorer = _extract_js_function(js, "initClassSplitExplorer")
    start_analysis = _extract_js_function(js, "startClassSplitAnalysis")
    reset_annotation_source = _extract_js_function(js, "resetAnnotationSourceState")

    assert "classSplitElements.trainingCapture.checked = false;" in reset_capture
    assert "classSplitElements.trainingCapture.checked = false;" in init_explorer
    assert "resetClassSplitTrainingCapture(" in start_analysis
    assert "turned off for the new analysis job" in start_analysis
    assert "resetClassSplitTrainingCapture(" in reset_annotation_source
    assert "turned off because the annotation source changed" in reset_annotation_source
    assert "localStorage" not in reset_capture


def test_desktop_confirm_skip_and_undo_forward_consent_and_review_provenance():
    js = _read(DESKTOP_JS_PATH)
    save_disposition = _extract_js_function(js, "saveClassSplitReviewDisposition")

    assert "...CLASS_SPLIT_REVIEW_DISPOSITIONS" in save_disposition
    assert '"clear"' in save_disposition
    assert "const captureTrainingData = options.captureTrainingData === undefined" in save_disposition
    assert "? isClassSplitTrainingCaptureEnabled()" in save_disposition
    assert "capture_training_data: captureTrainingData," in save_disposition
    assert "reviewDispositionRetryTokens.get(" in save_disposition
    assert "clientActionId = createClassSplitTrainingClientActionId();" in save_disposition
    assert "requestPayload.client_action_id = clientActionId;" in save_disposition
    assert "const reviewId = getClassSplitPointReviewId(safeId);" in save_disposition
    assert "requestPayload.review_id = reviewId;" in save_disposition
    assert 'safeDisposition === "confirm_current"' in save_disposition
    assert "pairDisposition" in save_disposition
    assert "requestPayload.dual_bbox_conflict" in save_disposition
    assert "getClassSplitAnnotationTarget(" in save_disposition
    assert (
        "requestPayload.annotation_target = annotationTarget;"
        in save_disposition
    )
    assert "body: JSON.stringify(requestPayload)" in save_disposition

    skip = _extract_js_function(js, "skipClassSplitWrongCandidate")
    confirm = _extract_js_function(js, "markClassSplitWrongCandidateCorrect")
    undo = _extract_js_function(js, "restoreClassSplitReviewDisposition")
    assert 'saveClassSplitReviewDisposition(safeId, "skip")' in skip
    assert 'saveClassSplitReviewDisposition(safeId, "confirm_current")' in confirm
    assert 'saveClassSplitReviewDisposition(safeId, "clear", {' in undo
    assert "clearPrecondition," in undo


def test_desktop_discard_is_canonical_queue_navigation_and_capture_is_nonblocking():
    js = _read(DESKTOP_JS_PATH)
    discard = _extract_js_function(js, "discardFirstClassSplitWrongCandidates")

    assert 'action_type: "discard"' in discard
    assert "point_ids: discardedIds" in discard
    assert 'origin: "desktop"' in discard
    assert "client_action_id: clientActionId" in discard
    assert "capturePayload.review_id =" in discard
    assert "capturePayload.review_ids =" in discard

    # Discard is queue navigation, not an annotation decision.
    for forbidden in ("before_class:", "after_class:", "target_class:", "label_commit_status:"):
        assert forbidden not in discard

    dismiss_at = discard.index("classSplitState.dismissedWrongIds.add(pointId)")
    rerender_at = discard.index("renderClassSplitWrongList();", dismiss_at)
    capture_at = discard.index("void captureClassSplitTrainingAction(")
    assert dismiss_at < rerender_at < capture_at


def test_desktop_class_change_uses_pending_then_committed_two_phase_capture():
    js = _read(DESKTOP_JS_PATH)
    change_class = _extract_js_function(js, "changeClassSplitPointClass")
    annotation_target = _extract_js_function(
        js, "getClassSplitAnnotationTarget"
    )

    flush_at = change_class.index(
        "await flushAnnotationSnapshot({ manual: false })"
    )
    committed_at = change_class.index('labelCommitStatus = "committed";', flush_at)
    pending_status_at = change_class.index(
        'label_commit_status: "pending_desktop_sync"'
    )
    pending_capture_at = change_class.index(
        "const pendingResult = await captureClassSplitTrainingAction(",
        pending_status_at,
    )
    commit_type_at = change_class.index(
        'action_type: "commit_class_change"',
        pending_capture_at,
    )
    queue_at = change_class.index(
        "pendingTrainingCommitKey = queueClassSplitPendingTrainingCommit({",
        commit_type_at,
    )
    drain_at = change_class.index(
        "commitRecorded = await drainClassSplitPendingTrainingCommits({",
        queue_at,
    )
    assert (
        flush_at
        < committed_at
        < pending_status_at
        < pending_capture_at
        < commit_type_at
        < queue_at
        < drain_at
    )

    assert 'action_type: "change_class"' in change_class
    assert "before_class: beforeClass" in change_class
    assert "after_class: targetClass" in change_class
    assert "client_action_id: clientActionId" in change_class
    assert "pendingPayload.review_id = reviewId;" in change_class
    assert "pendingPayload.annotation_target = annotationTarget;" in change_class
    assert "commitPayload.annotation_target = annotationTarget;" in change_class
    assert (
        "commitPayload.annotation_before_revision = ("
        in change_class
    )
    assert "beforeAnnotationRevision" in change_class
    assert "commitPayload.annotation_commit_revision = String(" in change_class
    assert "commitPayload.annotation_source_identity = String(" in change_class
    assert 'annotationSourceState.mode || ""' in annotation_target
    assert "annotationSourceState.datasetId" in annotation_target
    assert "annotationSourceState.sessionId" in annotation_target
    assert "row.image_relpath" in annotation_target
    assert "commits_action_id: pendingTrainingActionId" in change_class
    assert "commitPayload.review_id = reviewId;" in change_class
    assert "labelPersisted: false" in change_class
    assert "remains ineligible until its label save succeeds" in change_class


def test_optional_desktop_capture_failure_is_visible_without_reversing_the_action():
    js = _read(DESKTOP_JS_PATH)
    capture_action = _extract_js_function(js, "captureClassSplitTrainingAction")
    discard = _extract_js_function(js, "discardFirstClassSplitWrongCandidates")

    assert "if (!enabled)" in capture_action
    assert "capture_training_data: true" in capture_action
    assert "Review action completed, but future training capture failed" in capture_action
    assert 'setClassSplitTrainingCaptureStatus(message, "warn")' in capture_action
    assert "setSamStatus(message, { variant: \"warn\"" in capture_action
    assert "return null;" in capture_action

    # A capture exception cannot undo queue navigation: the UI action happens
    # first and the optional request is deliberately fire-and-forget.
    assert discard.index("classSplitState.dismissedWrongIds.add(pointId)") < discard.index(
        "void captureClassSplitTrainingAction("
    )


def test_committed_desktop_class_capture_is_awaited_and_kept_alive():
    js = _read(DESKTOP_JS_PATH)
    change_class = _extract_js_function(js, "changeClassSplitPointClass")
    capture_action = _extract_js_function(js, "captureClassSplitTrainingAction")
    drain_commits = _extract_js_function(js, "drainClassSplitPendingTrainingCommits")

    assert "const pendingResult = await captureClassSplitTrainingAction(" in change_class
    assert "await drainClassSplitPendingTrainingCommits({" in change_class
    assert "entry.labelPersisted !== true" in drain_commits
    assert "const result = await captureClassSplitTrainingAction(" in drain_commits
    assert "classSplitTrainingCaptureRecorded(result)" in drain_commits
    assert "pendingTrainingClassCommits.delete(entry.queueKey)" in drain_commits
    assert "result?.retryable === false" in drain_commits
    assert "permanently rejected" in drain_commits
    assert "const captureJobId = String(classSplitState.currentJobId ||" in change_class
    assert "jobId: captureJobId" in change_class
    assert "keepalive: true" in capture_action
    assert "retryable: ![400, 404, 409, 410, 413, 422].includes(" in capture_action


def test_combined_desktop_review_disposition_request_is_kept_alive():
    js = _read(DESKTOP_JS_PATH)
    save_disposition = _extract_js_function(js, "saveClassSplitReviewDisposition")

    assert "capture_training_data: captureTrainingData" in save_disposition
    assert "keepalive: true" in save_disposition
